from dataloaders.loaders import (
    crop_train_loader,
    weather_train_loader
)


def main():

    print("=" * 60)
    print("TESTING CROP DATALOADER")
    print("=" * 60)

    images, labels = next(iter(crop_train_loader))

    print("Image tensor shape :", images.shape)
    print("Label tensor shape :", labels.shape)
    print("Labels             :", labels[:10])


    print("\n" + "=" * 60)
    print("TESTING WEATHER DATALOADER")
    print("=" * 60)

    images, labels = next(iter(weather_train_loader))

    print("Image tensor shape :", images.shape)
    print("Label tensor shape :", labels.shape)
    print("Labels             :", labels[:10])


if __name__ == "__main__":
    main()