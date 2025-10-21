from fastapi import FastAPI
import random
import csv

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

# Load hospitals from CSV
hospitals = []
with open("hospitals.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hospitals.append({
            "name": row["name"],
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "facilities": row.get("facilities", "")
        })

@app.get("/hospitals")
def get_hospitals():
    hospital_status = []
    for h in hospitals:
        total_beds = random.randint(200, 800)
        icu_beds = random.randint(5, 100)
        emergency_load = random.choice(["Low", "Medium", "High"])
        capacity_utilization = round(random.uniform(0.3, 0.95), 2)

        hospital_status.append({
            "name": h["name"],
            "lat": h["lat"],
            "lon": h["lon"],
            "total_beds": total_beds,
            "icu_beds": icu_beds,
            "emergency_load": emergency_load,
            "capacity_utilization": capacity_utilization,
            "facilities": h["facilities"]
        })
    return {"hospitals": hospital_status}
