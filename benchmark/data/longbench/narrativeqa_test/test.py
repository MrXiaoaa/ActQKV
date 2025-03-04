import pyarrow as pa

# Load the dataset from the specified file using pa.ipc.open_file
dataset_path = './benchmark/data/longbench/narrativeqa_test/data-00000-of-00001.arrow'
with pa.memory_map(dataset_path, 'r') as source:
    reader = pa.ipc.open_file(source)
    table = reader.read_all()

# Optionally, you can print the schema of the dataset to understand its structure
print("Dataset Schema:")
print(table.schema)

# Take the first 5 rows
table = table.slice(0, 5)

# Save the first 5 rows back to a new Arrow file
output_path = './benchmark/data/longbench/narrativeqa_test/data-00000-of-00001-truncated.arrow'
with pa.OSFile(output_path, 'wb') as sink:
    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(table)

# Optionally, print the first 5 rows to verify
print("First 5 rows:")
print(table)
