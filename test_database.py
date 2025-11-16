from backend.db import SessionLocal, Run, init_db, RunChunk, AttributionRow

# Make sure tables exist
init_db()

db = SessionLocal()
print("Runs:", db.query(Run).count())
for r in db.query(Run).all():
    print(r.run_id, r.user_query, r.selection_mode)
db.close()
for r in db.query(Run).all():
    print(r.run_id, r.user_query, r.selection_mode)

print("RunChunks:", db.query(RunChunk).count())
print("Attributions:", db.query(AttributionRow).count())
db.close()