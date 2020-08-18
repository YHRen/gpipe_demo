.PHONY: tar 

tar: main.py readme.md *.sh Makefile
	tar cfvz gpipe_demo.tar.gz $?

scp: gpipe_demo.tar.gz
	scp $< ascent:~/stlearn/gpipe_demo/

