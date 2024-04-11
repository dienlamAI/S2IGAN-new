from data.dataset import SENDataset


def main():
    dataset = SENDataset(
        r"C:\Users\nvatu\OneDrive\Desktop\s2idata\train_flower_en2vi.json",
        r"C:\Users\nvatu\OneDrive\Desktop\s2idata\image_oxford\image_oxford\train",
        r"C:\Users\nvatu\OneDrive\Desktop\s2idata\oxford\oxford\train",
    )


if __name__ == "__main__":
    main()
