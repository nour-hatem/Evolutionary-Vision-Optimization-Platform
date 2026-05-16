# Cloud Deployment — Commands Cheat Sheet

> **Replace all `<EC2_IP>` placeholders with your actual EC2 instance public IP.**  
> Set `DATABASE_URL` in your environment (`.env` file) — never hardcode credentials.

---

## Connect to EC2

```bash
ssh -i <your-pem-key>.pem ubuntu@<EC2_IP>
```

---

## Health Checks

```bash
# Check all containers are running
docker ps

# Check API health + DB connection
curl http://localhost:8000/health

# Check memory usage
free -h
```

---

## View Prediction Logs (from UI usage)

```bash
docker exec cifar-api python -c "
from experiments.db_models import SessionLocal, PredictionLog
db = SessionLocal()
header = '{:<5} {:<20} {:<15} {:<10} {:<22}'.format('ID','Filename','Class','Conf','Time')
print(header)
print('-'*75)
for p in db.query(PredictionLog).all():
    print('{:<5} {:<20} {:<15} {:<10} {:<22}'.format(
        p.id, str(p.filename)[:20], p.predicted_class,
        round(p.confidence,4), str(p.timestamp)))
db.close()
"
```

---

## View Experiments

```bash
docker exec cifar-api python -c "
from experiments.db_models import SessionLocal, Experiment
db = SessionLocal()
header = '{:<5} {:<20} {:<12} {:<26} {:<26}'.format('ID','Run Name','Status','Started','Ended')
print(header)
print('-'*90)
for e in db.query(Experiment).all():
    print('{:<5} {:<20} {:<12} {:<26} {:<26}'.format(
        e.id, e.run_name, e.status, str(e.start_time), str(e.end_time)))
db.close()
"
```

---

## View Run Logs (GA Training)

```bash
docker exec cifar-api python -c "
from experiments.db_models import SessionLocal, RunLog
db = SessionLocal()
header = '{:<7} {:<5} {:<12} {:<12} {:<22}'.format('ExpID','Gen','Accuracy','F1','Timestamp')
print(header)
print('-'*60)
for l in db.query(RunLog).all():
    print('{:<7} {:<5} {:<12} {:<12} {:<22}'.format(
        l.experiment_id, l.generation, str(l.accuracy), str(l.f1_score), str(l.timestamp)))
db.close()
"
```

---

## Start a New Experiment (Live Demo)

```bash
curl -X POST http://localhost:8000/start_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "demo_run_1", "max_generations": 3, "population_size": 5, "mutation_rate": 0.3, "dataset": "cifar10"}'
```

---

## Checkpoint Resume Demo (Spot Interruption Simulation)

**Step 1 — Start experiment with 6 generations, interrupt after 3:**
```bash
curl -X POST http://localhost:8000/start_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "spot_demo_1", "max_generations": 6, "population_size": 8, "mutation_rate": 0.3, "dataset": "cifar10", "interrupt_after": 3}'
```

**Step 2 — Wait, then check status (should show "interrupted" + checkpoint at gen 2):**
```bash
sleep 5
curl -s http://localhost:8000/experiment_status/spot_demo_1 | python3 -m json.tool
```

**Step 3 — Resume from checkpoint:**
```bash
curl -X POST http://localhost:8000/resume_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "spot_demo_1"}'
```

**Step 4 — Verify completion (should show "completed" with all 6 generations):**
```bash
sleep 5
curl -s http://localhost:8000/experiment_status/spot_demo_1 | python3 -m json.tool
```

---

## Predict from Terminal (without UI)

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/image.jpg"
```

---

## View All DB Tables

```bash
docker exec cifar-api python -c "
from experiments.db_models import engine
from sqlalchemy import inspect
i = inspect(engine)
for t in i.get_table_names():
    cols = [c['name'] for c in i.get_columns(t)]
    print(f'{t}: {cols}')
"
```

---

## Export to CSV

```bash
# Export Experiments
docker exec cifar-api python -c "
import csv, sys
from experiments.db_models import SessionLocal, Experiment
db = SessionLocal()
w = csv.writer(sys.stdout)
w.writerow(['id','run_name','status','start_time','end_time'])
for e in db.query(Experiment).all():
    w.writerow([e.id, e.run_name, e.status, e.start_time, e.end_time])
db.close()
" > ~/experiments.csv

# Export Run Logs
docker exec cifar-api python -c "
import csv, sys
from experiments.db_models import SessionLocal, RunLog
db = SessionLocal()
w = csv.writer(sys.stdout)
w.writerow(['experiment_id','generation','accuracy','f1_score','timestamp'])
for l in db.query(RunLog).all():
    w.writerow([l.experiment_id, l.generation, l.accuracy, l.f1_score, l.timestamp])
db.close()
" > ~/run_logs.csv

# Export Prediction Logs
docker exec cifar-api python -c "
import csv, sys
from experiments.db_models import SessionLocal, PredictionLog
db = SessionLocal()
w = csv.writer(sys.stdout)
w.writerow(['id','filename','predicted_class','confidence','timestamp'])
for p in db.query(PredictionLog).all():
    w.writerow([p.id, p.filename, p.predicted_class, p.confidence, p.timestamp])
db.close()
" > ~/prediction_logs.csv
```

---

## Container Management

```bash
docker restart cifar-api
docker restart cifar-ui
docker logs cifar-api --tail 20
docker logs cifar-ui --tail 20
docker stop cifar-api cifar-ui
docker start cifar-api cifar-ui
```

---

## Emergency: Instance Unresponsive

1. Go to **AWS Console → EC2 → Instances**
2. Select instance → **Instance State → Reboot**
3. Wait 1-2 minutes, then SSH in with the **new IP**
4. Run `docker ps` to verify containers auto-restarted

---

## Quick Demo Flow

1. **Open UI** → `http://<EC2_IP>:8501` — show the dashboard
2. **Open Swagger** → `http://<EC2_IP>:8000/docs` — show API docs
3. **Run prediction from UI** → upload an image, show result
4. **Check DB** → run "View Prediction Logs" command above
5. **Start experiment** → run "Start a New Experiment" command
6. **Show experiment logged** → run "View Experiments" command
7. **Show training logs** → run "View Run Logs" command
