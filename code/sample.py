from sparkdl import DeepImageFeaturizer, readImages


def main():
    normal_path = '../images/processed/sample'
    normal_df = readImages(normal_path)
    normal_df.show()
    normal_df.printSchema()
    print(normal_df)

if __name__ == "__main__":
    main()