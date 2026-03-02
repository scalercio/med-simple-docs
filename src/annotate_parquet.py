from tqdm.autonotebook import tqdm
import pandas as pd
from collections import defaultdict 
import sys
import pandas as pd
from metricas_simplificacao import annotate_list

parquet_file = sys.argv[1]
df = pd.read_parquet(parquet_file)
#original_docs = df['docs'].tolist()
simple_docs = df['simple_doc'].tolist()
#annotate_list(original_docs, parquet_file + "_original")
annotate_list(simple_docs, parquet_file + "_simple")
