import datasets
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    path = "/".join(args.output_dir.split("/")[:-1])
    if not os.path.exists(path):
        os.makedirs(path)

    mnli = datasets.load_dataset("multi_nli")["train"]
    snli = datasets.load_dataset("snli")["train"]

    sents = mnli["premise"]+mnli["hypothesis"]+snli["premise"]+snli["hypothesis"]
    sents = list(dict.fromkeys(sents))
    with open(args.output_dir,"w") as output_h:
        print("text", file=output_h)
        for i, line in enumerate(sents):
            if i%1001==0:
                print(".", end="")
            elif i%10001==0:
                print(f"{i-1}", end="")
            print(line, file=output_h)
    print("done")

if __name__=="__main__":
    main()