import os

source_root = 'D:\\1092\\ML\\school_building'

for dirPath, dirNames, fileNames in os.walk(source_root):
    for dirName in dirNames:
        for dP, dN, fileNames in os.walk(dirName):
            count = 0
            for fileName in fileNames:
                # rename file
                newName = dirName + '_' + str(count) + '.JPG'
                count += 1
                oldPath = dirPath+'\\'+dirName+'\\'+ fileName
                newPath = dirPath+'\\'+dirName+'\\'+ newName
                os.rename(oldPath, newPath)