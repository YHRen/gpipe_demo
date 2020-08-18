.PHONY: tar 

tar: main.py readme.md *.sh Makefile
        tar cfvz gpipe_demo.tar.gz $?

