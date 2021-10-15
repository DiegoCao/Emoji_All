import json

with open("./datasample/2018-01-01-0.json", "r") as fp:
    line = fp.readline()

    while line:
        js = json.loads(line)
        print(json.dumps(js, indent=2, sort_keys=True))
        a = input()
        line = fp.readline()

