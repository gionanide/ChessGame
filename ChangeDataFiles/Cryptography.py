from Crypto.PublicKey import RSA
from Crypto import Random

#make a random key
random_generator = Random.new().read
key = RSA.generate(1024, random_generator)
print('Your random key for RSA algorithm is:')
print(key)
#First, we extract the public key from the key pair and use it to encrypt some data
public_key = key.publickey()
# 32 is a random parameter used by the RSA algorithm to encrypt the data
print('Encryption starts')
with open('new.txt','r') as inputFile:
    with open('RSAencrypt.txt','w') as rsaFile:
        for line in inputFile.readlines():
            #encrypt the data with RSA algorithm
            encryptedData = public_key.encrypt(line, 32)
            #make the encrypted data to a String in order to wirte them into the file
            newLine = str(encryptedData)
            #write in the encrypted file
            rsaFile.write(newLine)
            break

#Decryption procedure
with open('RSAencrypt.txt','r') as inFile:
    with open('Decrypt.txt','w') as decrypt:
        for line in inFile.readlines():
            dec = key.decrypt(line)
            newLine1 = str(dec)
            decrypt.write(newLine1)





