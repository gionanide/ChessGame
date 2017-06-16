import json
#make your data in json format
with open("new.txt", "r") as file:
    with open('output.txt', 'w') as outFile:
        for line in file.readlines():
            y=json.dumps(line)#make line json
            outFile.write(y)#write your line in the outputfile

#view your new file
with open('output.txt','r') as oF:
    for line in oF.readlines():
        print(line)

