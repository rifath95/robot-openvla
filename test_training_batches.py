"""Check measured throughput and conservative memory screening without a GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_training_batches import summarize,write_report,next_capacity_batch,missed_speed_candidates
from training.integrate_upstream import integrated_source
from summarize_gpu_benchmarks import comparison,duration_line


class BatchBenchmarkTests(unittest.TestCase):
    def rows(self):
        return [dict(update=i,elapsed_seconds=100+i*2,train_loss=1,
                     gpu_peak_allocated_gib=18,gpu_peak_reserved_gib=27,
                     gpu_free_gib=17,gpu_total_gib=44) for i in range(1,26)]

    def test_throughput_excludes_warmup_and_final_validation(self):
        rows=self.rows()+[dict(update=25,elapsed_seconds=999,validation_loss=2)]
        result=summarize(rows,2,2)
        self.assertEqual(result['measured_updates'],20)
        self.assertEqual(result['seconds_per_update'],2)
        self.assertEqual(result['examples_per_second'],2)
        self.assertTrue(result['headroom_screen_passed'])

    def test_reserved_and_external_memory_can_each_fail_screen(self):
        for key,value in [('gpu_peak_reserved_gib',42),('gpu_free_gib',2)]:
            rows=self.rows();rows[-1][key]=value
            self.assertFalse(summarize(rows,4,1)['headroom_screen_passed'])

    def test_report_excludes_oom_and_unsafe_candidates_from_selection(self):
        safe=dict(summarize(self.rows(),2,2),status='completed')
        unsafe=dict(safe,microbatch=4,examples_per_second=100,headroom_screen_passed=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_report(root,[dict(microbatch=1,accumulation=4,status='out_of_memory'),safe,unsafe])
            report=(root/'BATCH_BENCHMARK.md').read_text()
            self.assertIn('microbatch 2, accumulation 2.',report)
            self.assertEqual(len(json.loads((root/'benchmark.json').read_text())),3)

    def test_metrics_follow_optimizer_and_precede_validation(self):
        source=integrated_source()
        self.assertLess(source.index('optimizer.step()'),source.index('wandb.log('))
        self.assertLess(source.index('wandb.log('),source.index('validate(vla'))

    def test_adaptive_search_finds_non_power_of_two_boundary(self):
        results=[];tried=[]
        while (batch:=next_capacity_batch(results)) is not None:
            tried.append(batch)
            results.append(dict(microbatch=batch,status='completed' if batch<=6 else 'out_of_memory'))
        self.assertEqual(tried,[1,2,4,8,6,7])
        self.assertEqual(max(r['microbatch'] for r in results if r['status']=='completed'),6)

    def test_search_stops_at_bound_or_when_even_one_fails(self):
        self.assertIsNone(next_capacity_batch([dict(microbatch=1,status='out_of_memory')]))
        self.assertIsNone(next_capacity_batch([dict(microbatch=8,status='completed')],maximum=8))
        self.assertEqual(next_capacity_batch([dict(microbatch=4,status='completed')],maximum=6),6)
        with self.assertRaises(ValueError):
            next_capacity_batch([dict(microbatch=1,status='failed')])

    def test_speed_refinement_fills_only_missed_fitting_sizes(self):
        results=[]
        while (batch:=next_capacity_batch(results)) is not None:
            results.append(dict(microbatch=batch,status='completed' if batch<=13 else 'out_of_memory'))
        self.assertEqual(missed_speed_candidates(results),[9,10,11])
        self.assertEqual(missed_speed_candidates([dict(microbatch=1,status='out_of_memory')]),[])

    def test_bound_is_reported_as_lower_bound_not_maximum(self):
        row=dict(summarize(self.rows(),8,1),status='completed')
        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory),[row],mode='capacity',maximum=8)
            report=(Path(directory)/'BATCH_BENCHMARK.md').read_text()
            self.assertIn('the maximum was not established',report)
            self.assertIn('effective batch changes',report)

    def test_comparison_retains_gpus_and_uses_safe_candidate_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'runs';output=Path(directory)/'comparison.md'
            for name,gpu,vram,mode in [('a40','NVIDIA A40',44,'capacity'),
                                      ('a100','NVIDIA A100-SXM4-80GB',80,'controlled')]:
                run=root/name;run.mkdir(parents=True)
                safe=dict(summarize(self.rows(),2,1),status='completed')
                unsafe=dict(safe,microbatch=6,effective_batch=6,examples_per_second=99,
                            headroom_screen_passed=False)
                rows=[safe,unsafe,dict(microbatch=7,accumulation=1,status='out_of_memory')]
                (run/'benchmark.json').write_text(json.dumps(rows))
                (run/'benchmark_metadata.json').write_text(json.dumps(dict(mode=mode,dataset='/workspace/training_v1',benchmark_elapsed_seconds=702.49,benchmark_outcome='completed')))
                candidate=run/'batch_2_accum_1';candidate.mkdir()
                (candidate/'model_info.json').write_text(json.dumps(dict(gpu=gpu,total_vram_gib=vram)))
            self.assertEqual(comparison(root,output),2)
            text=output.read_text()
            self.assertIn('NVIDIA A40',text)
            self.assertIn('NVIDIA A100-SXM4-80GB',text)
            self.assertIn('6 (boundary found)',text)
            self.assertIn('≥6 (fixed candidates only)',text)
            self.assertIn('| 2 / 1 | 2.000 | 1.00 |',text)
            # Summary uses the safe candidate; details still show the faster
            # unsafe batch so readers can understand the tradeoff.
            self.assertIn('99.00',text)
            self.assertIn('## NVIDIA A40',text)
            self.assertIn('## NVIDIA A100-SXM4-80GB',text)
            self.assertIn('Capacity finding:',text)
            self.assertEqual(text.count('Total benchmark time: **11 min 42 sec**.'),2)
            self.assertEqual(comparison(root,output,gpu_filter='NVIDIA A40'),1)
            self.assertNotIn('NVIDIA A100-SXM4-80GB',output.read_text())

    def test_total_duration_uses_full_timer_or_all_legacy_tests_including_oom(self):
        rows=[dict(wall_seconds=167.24),dict(wall_seconds=535.25,status='out_of_memory')]
        self.assertIn('11 min 42 sec',duration_line(rows,{}))
        self.assertIn('sum of batch-test wall times',duration_line(rows,{}))
        self.assertEqual(duration_line(rows,dict(benchmark_elapsed_seconds=720,benchmark_outcome='completed')),
                         'Total benchmark time: **12 min 00 sec**.')
        self.assertIn('so far',duration_line(rows,dict(benchmark_elapsed_seconds=720,benchmark_outcome='running')))
        self.assertIn('not recorded',duration_line([{}],{}))

    def test_empty_comparison_does_not_invent_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'comparison.md'
            self.assertEqual(comparison(Path(directory)/'missing',output),0)
            self.assertIn('Not measured',output.read_text())


if __name__=='__main__':unittest.main()
