uname -sr
python3 --version
printf '3\n1 2 3\n' | python3 -I -S -c 'import os;n=os.fstat(0).st_size;print("pipe st_size",n,len(os.read(0,n)))'
printf '3\n1 2 3\n' > /tmp/cd_in.txt
python3 -I -S -c 'import os;n=os.fstat(0).st_size;print("file st_size",n,len(os.read(0,n)))' < /tmp/cd_in.txt
python3 -I -S -c 'exit()' 2>/dev/null; echo "rc bare exit under -S: $?"
python3 -I -c 'exit()'; echo "rc bare exit without -S: $?"
