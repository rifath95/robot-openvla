"""Append persistent stage timings even when a stage raises or a job is interrupted."""
from contextlib import contextmanager
from datetime import datetime,timezone
import json,time
from pathlib import Path


def utc():return datetime.now(timezone.utc).isoformat()


def read_events(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        try:row=json.loads(line)
        except ValueError:continue  # A killed writer can leave its final event incomplete.
        if isinstance(row,dict) and row.get('event') in ('started','finished'):rows.append(row)
    return rows


class Timings:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'stage_timings.jsonl'
    def event(self,record):
        with self.path.open('ab+') as stream:
            stream.seek(0,2)
            if stream.tell():
                stream.seek(-1,2)
                if stream.read(1)!=b'\n':stream.seek(0,2);stream.write(b'\n')
            stream.seek(0,2);stream.write((json.dumps(record)+'\n').encode());stream.flush()
    @contextmanager
    def stage(self,name,**details):
        start=time.monotonic();at=utc();self.event(dict(stage=name,event='started',at_utc=at,**details))
        result='completed';error=None
        try:yield
        except BaseException as exc:
            result='failed';error=type(exc).__name__;raise
        finally:
            self.event(dict(stage=name,event='finished',started_at_utc=at,finished_at_utc=utc(),seconds=time.monotonic()-start,status=result,error=error,**details))
            self.report()
    def report(self):
        rows=read_events(self.path)
        finished=[r for r in rows if r['event']=='finished']
        lines=['# Chained run timings','','Wall times include retries inside each stage. Nested model/trial records are not added to the top-level total.','',
               '| Stage | Status | Wall seconds | Started UTC | Finished UTC |','|---|---|---:|---|---|']
        for r in finished:lines.append(f'| {r["stage"]} | {r["status"]} | {r["seconds"]:.2f} | {r["started_at_utc"]} | {r["finished_at_utc"]} |')
        pending={r['stage'] for r in rows if r['event']=='started'}-{r['stage'] for r in finished}
        if pending:lines+=['',f'Not yet finished: {", ".join(sorted(pending))}']
        (self.directory/'TIMINGS.md').write_text('\n'.join(lines)+'\n')
