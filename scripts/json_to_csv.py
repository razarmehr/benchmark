import pandas as pd
import json
import argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('file',type=argparse.FileType('r'))
parser.add_argument('--device',type=str, default='mps', help="Device you want to set such as \'cuda\' or \'mps\'")
args = parser.parse_args()

data = json.load(args.file)
df = pd.DataFrame(data["benchmarks"])
df.to_csv('out.csv', encoding='utf-8', index=False)
df = pd.read_csv('out.csv', sep = ',')
print (df.head())
df['test_type'] = df['name'].str.split("[").str.get(0).str.split("_").str.get(1)
df['test_category'] =  df['name'].str.split("[").str.get(1).str.split("-").str.get(-2)
df['test_device'] =  df['param'].str.split("-").str.get(1)
df['test_suffix'] = df['name'].str.split("[").str.get(1).str.split("-").str.get(-1).str[:-1]
df['test_name'] =  df['name'].str.split("[").str.get(1).str.split("-").str.get(0)
df['test_name'] = df['test_name'] + "-" + df['test_suffix']
#df = pd.concat([df, df["stats"].apply(pd.Series)], axis=1)
df["stats_1"] = df["stats"].apply(lambda x : dict(eval(x)) )

df = pd.concat([df, df["stats_1"].apply(pd.Series )], axis = 1)
# df
# 
df[['name', 'test_type', 'test_name', 'test_device', 'median']].head(30)
df_mps = df[df['test_device'].isin([args.device, 'cpu'])]
df_mps['combined'] = df_mps['test_type']+"_"+df_mps['test_category']+"_"+df_mps['test_device']
df_mps = df_mps.pivot(index = 'test_name', columns = ['combined'], values = 'median').reset_index()
for item in ['eval_eager', 'eval_jit', 'train_eager', 'train_jit' ]:
    new_col = "ratio_cpu_" + args.device + "_"+item
    if (item+"_cpu") in df_mps.columns and (item+"_"+args.device) in df_mps.columns:
        df_mps[new_col] = df_mps[item+"_cpu"]/df_mps[item+"_"+args.device]
print (df_mps)

def check_cols(df_mps, col):
    print (df_mps.columns)
    if (col in df_mps.columns):
        return (~(pd.isnull(df_mps[col])))
    else:
        return True

print(df_mps.shape)
#print(df_mps.columns.tolist())
for colname in df_mps.columns.tolist()[1:]:
    print(colname)
    df_mps = df_mps[~(pd.isnull(df_mps[colname]))]

    # df_mps = df_mps[check_cols(df_mps,('ratio_cpu_' + args.device + '_eval_eager')) |
    #             check_cols(df_mps,('ratio_cpu_' + args.device + '_eval_jit')) |
    #             check_cols(df_mps,('ratio_cpu_' + args.device + '_train_eager')) |
    #             check_cols(df_mps,('ratio_cpu_' + args.device + '_train_jit'))]
print(df_mps.shape)
df_mps.to_csv("out.csv",encoding='utf-8', index=False)
