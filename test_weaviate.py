import weaviate

# Connect to Weaviate
client = weaviate.connect_to_local()

# Get collection
collection = client.collections.get("LawArticles")

# Count total documents
print(f"Total documents: {len(collection)}")

# Get first 5 documents
print("\n📄 First 5 documents:")
print("=" * 80)

response = collection.query.fetch_objects(limit=5)
for item in response.objects:
    print(f"\n🔹 Law: {item.properties.get('law')}")
    print(f"   Article ID: {item.properties.get('article_id')}")
    print(f"   Title: {item.properties.get('title')}")
    print(f"   Chapter: {item.properties.get('chapter')}")
    print(f"   Text: {item.properties.get('text')[:100]}...")
    print(f"   Source: {item.properties.get('source_file')}")
    print(f"   Granularity: {item.properties.get('granularity')}")

# Get schema/properties
print("\n\n📋 Collection Schema:")
print("=" * 80)
schema = collection.config.get()
print(f"Collection name: {schema.name}")
print(f"Properties:")
for prop in schema.properties:
    print(f"  - {prop.name}: {prop.data_type}")

client.close()
print("\n✅ Done!")
