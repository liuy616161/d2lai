import csv
csv_file=open('label.csv','w')
writer=csv.writer(csv_file)
writer.writerow(['data','label'])
for i in range (0,55):
    writer.writerow(['6.class/%02.f.jpg'%i,'6'])

for i in range (0,55):
    writer.writerow(['1.class/%02.f.jpg'%i,'1'])
csv_file.close()
