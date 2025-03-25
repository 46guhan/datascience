#create a new file
#open("file.txt",'x')

#write

# f=open("file.txt",'w')
# f.write("that is something!!!!!")
# f.close()

#append
# f=open("file.txt",'a')
# f.write("\tthis is new element")
# f.close()

#read
""" f=open("file.txt",'r')
d=f.readlines()
for x in d:
    x1=x.split(" ")
    print("Name : "+x1[0])
    print("age : "+x1[1]) """


""" f=open("file.txt",'a')
f.write("\nbalaji\t31\tjava developer\t20000\t123456")
f.write("\nmanoj\t41\tjava developer\t20000\t123456")
f.write("\nvarshini\t22\tjava developer\t20000\t123456")
f.close() """

name=input("Enter the Username")
password=input("Enter the Password")

f=open("file.txt",'r')
data=f.readlines()
for line in data:
    ud=line.split("\n")
    print(ud)
  
f.close()