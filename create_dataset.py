import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.itertools import Map
from datasets import Dataset, Features, Value, Sequence

url = (
    "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
    "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)
df.head()

ds = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

class ProcessStartField():
    ts_id = 0
    
    def __call__(self, data):
        data["start"] = data["start"].to_timestamp()
        data["feat_static_cat"] = [self.ts_id]
        self.ts_id += 1
        
        return data

process_start = ProcessStartField()

list_ds = list(Map(process_start, ds))

features  = Features(
    {    
        "start": Value("timestamp[s]"),
        "target": Sequence(Value("float32")),
        "feat_static_cat": Sequence(Value("uint64")),
        # "feat_static_real":  Sequence(Value("float32")),
        # "feat_dynamic_real": Sequence(Sequence(Value("uint64"))),
        # "feat_dynamic_cat": Sequence(Sequence(Value("uint64"))),
        "item_id": Value("string"),
    }
)

dataset = Dataset.from_list(list_ds, features=features)
dataset
