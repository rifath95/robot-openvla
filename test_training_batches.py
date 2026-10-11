"""Check measured throughput and conservative memory screening without a GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_training_batches import summarize,write_report
from training.integrate_upstream import integrated_source


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


if __name__=='__main__':unittest.main()
