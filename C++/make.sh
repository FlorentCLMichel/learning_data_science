
if [ $# -eq 0 ]
	then
		echo Hi there! Which library would you like to compile?
		read -p "library name " libname
	else
		libname="$1"
fi
g++ -O3 -I $HOME/anaconda3/include/python3.7m -fpic -c -o $libname.o $libname.cpp
g++ -o $libname.so -shared $libname.o -lboost_python3
rm $libname.o
