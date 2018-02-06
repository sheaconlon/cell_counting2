FILESERVER=https://d1rq19kap4wht6.cloudfront.net
EASY_VERSION=1

all: easy
	
easy: clean-easy get-easy

get-easy:
	wget $(FILESERVER)/easy-v$(EASY_VERSION).zip
	unzip easy-v$(EASY_VERSION).zip
	rm -f easy-v$(EASY_VERSION).zip

clean-easy:
	rm -f easy.zip
	rm -rf easy