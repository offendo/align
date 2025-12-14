import pandas as pd

def generate_benchmark(df):
    columns = ["formal", "informal"]
    # Get any of the rows which we fixed up.
    fixed = df[df.fixed_prediction.apply(len) > 0][["fixed_prediction", "informal"]]
    fixed.columns = columns
    # Get the rows which were correct to start with. This means no fix, compiled correctly, and no notes.
    correct = df[(df.fixed_prediction.apply(len) == 0) & df.verified & (df.notes.apply(len) == 0)][["prediction", "informal"]]
    correct.columns = columns
    # get any extras we made
    extra_goods = df[["additional_valid", "informal"]].explode("additional_valid").dropna()
    extra_goods.columns = columns
    goods = pd.concat([fixed, correct, extra_goods])
    goods["label"] = "aligned"

    # And now get the negative examples. This is things where it didn't compile, or it did but it's wrong (i.e., we had a note for it)
    bads = df[(~df.verified) | (df.verified & (df.notes.apply(len) > 0))][["prediction", "informal"]]
    bads.columns = columns
    bads["label"] = "misaligned"
    # Next, any additional incorrect values we have
    extra_bads = df[["additional_incorrect", "informal"]].explode("additional_incorrect").dropna()
    extra_bads.columns = columns
    extra_bads["label"] = "misaligned"
    # finally return everything
    alls = pd.concat([goods, bads, extra_bads]).reset_index(drop=True).drop_duplicates()
    alls['formal'] = alls.formal.str.strip()
    return alls

df = pd.read_json("./mathatlas_30_and_translation_notes.json")
bench = generate_benchmark(df)
bench.to_json("benchmark.jsonl", lines=True, orient="records")

