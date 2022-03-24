
clean:
	rm -rf __pycache__
install:
	cp DE10_NANO.rbf /lib/firmware/
	dtbocfg.rb --install soc_system --dts soc_system.dts

uninstall:
	dtbocfg.rb --remove soc_system
	rm -f /lib/firmware/DE10_NANO.rbf
