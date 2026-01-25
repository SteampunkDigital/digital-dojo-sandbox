"""Quick MongoDB diagnostic script"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
db_name = os.getenv("MONGODB_DATABASE", "digital_dojo")

print(f"Connecting to: {uri}")
print(f"Database: {db_name}")
print()

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command('ping')
    print("Connected to MongoDB successfully!")
    print()

    print("All databases:")
    for name in client.list_database_names():
        print(f"  - {name}")
    print()

    db = client[db_name]
    print(f"Collections in '{db_name}':")
    collections = db.list_collection_names()
    if not collections:
        print("  (no collections yet)")
    for coll_name in collections:
        count = db[coll_name].count_documents({})
        print(f"  - {coll_name}: {count} documents")
        if count > 0 and count <= 5:
            print("    Recent documents:")
            for doc in db[coll_name].find().limit(3):
                doc_str = str(doc)[:100]
                print(f"      {doc_str}...")

except Exception as e:
    print(f"Error: {e}")
    print()
    print("Make sure MongoDB is running:")
    print("  mongod --dbpath <your-data-path>")
