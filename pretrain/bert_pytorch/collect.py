import json
import pandas as pd
from matplotlib import pyplot as plt
import os

def collect(result_dir, impute):
    print(result_dir)
    if not impute:
        mode = 'train'
        print("evaluate " + mode)
        with open(result_dir + mode + "_log.json") as f:
            result = json.load(f)
        avg_loss = []
        loss = []

        for line in result:
            avg_loss.append(line['avg_loss'])
            loss.append(line["loss"])

        plt.plot(avg_loss)
        plt.title(mode+'_avg_loss')
        plt.savefig(result_dir + mode + "_avg_loss.png")
        plt.close()
        plt.plot(loss)
        plt.title(mode+'loss')
        plt.savefig(result_dir + mode + "_loss.png")
        plt.close()

        with open(result_dir + mode + "_epoch.json") as f:
            result = json.load(f)

        mask_loss = []
        avg_loss = []


        for line in result:

            mask_loss.append(line["total_mask_rmse"])
            avg_loss.append(line["avg_loss"])

        train = pd.DataFrame({
            "avg_loss": avg_loss,
            "mask_rmse": mask_loss
        })

        train.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
        print(train)

        mode = 'test'
        print("evaluate " + mode)
        with open(result_dir + mode + "_log.json") as f:
            result = json.load(f)
        avg_loss = []
        loss = []

        for line in result:
            avg_loss.append(line['avg_loss'])
            loss.append(line["loss"])

        plt.plot(avg_loss)
        plt.title(mode+'_avg_loss')
        plt.savefig(result_dir + mode + "_avg_loss.png")
        plt.close()
        plt.plot(loss)
        plt.title(mode+'loss')
        plt.savefig(result_dir + mode + "_loss.png")
        plt.close()


        with open(result_dir + mode + "_epoch.json") as f:
            result = json.load(f)

        mask_loss = []
        avg_loss = []

        for line in result:
            mask_loss.append(line["total_mask_rmse"])
            avg_loss.append(line["avg_loss"])

        test = pd.DataFrame({

            "avg_loss": avg_loss,
            "mask_rmse": mask_loss
        })

        test.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
        print(test)
    else:
        mode = 'train'
        print("evaluate " + mode)
        with open(result_dir + mode + "_log.json") as f:
            result = json.load(f)
        avg_loss = []
        loss = []
        for line in result:
            avg_loss.append(line['avg_loss'])

        plt.plot(avg_loss)
        plt.title(mode+'_avg_loss')
        plt.savefig(result_dir + mode + "_avg_loss.png")
        plt.close()

        with open(result_dir + mode + "_epoch.json") as f:
            result = json.load(f)
        avg_loss = []
        mask_rmse = []

        for line in result:
            avg_loss.append(line["avg_loss"])
            mask_rmse.append(line["total_mask_rmse"])
        train = pd.DataFrame({
            "avg_loss": avg_loss,
            "mask_rmse":mask_rmse
        })
        train.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
        print(train)

        mode = 'test'
        print("evaluate " + mode)
        with open(result_dir + mode + "_log.json") as f:
            result = json.load(f)
        avg_loss = []
        loss = []
        for line in result:
            avg_loss.append(line['avg_loss'])

        plt.plot(avg_loss)
        plt.title(mode+'_avg_loss')
        plt.savefig(result_dir + mode + "_avg_loss.png")
        plt.close()

        with open(result_dir + mode + "_epoch.json") as f:
            result = json.load(f)
        avg_loss = []
        mask_rmse = []

        for line in result:
            avg_loss.append(line["avg_loss"])
            mask_rmse.append(line["total_mask_rmse"])
        test = pd.DataFrame({
            "avg_loss": avg_loss,
            "mask_rmse":mask_rmse
        })

        test.to_csv(result_dir+mode+"_epoch.tsv",sep="\t")
        print(test)
