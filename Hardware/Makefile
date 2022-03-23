ECHO := echo

.PHONY: all

all: rbf copy

qsys_edit:
	qsys-edit soc_system.qsys

rbf:
	@$(ECHO) "Running quartus flow"
	quartus_map DE10_NANO.qpf
	quartus_cdb --merge DE10_NANO.qpf
	quartus_stp DE10_NANO
	quartus_sh --flow compile DE10_NANO.qpf
	quartus_cpf -c -o bitstream_compression=on DE10_NANO.sof DE10_NANO.rbf
	sopc-create-header-files "./soc_system.sopcinfo" --single hps_0.h --module hps_0_arm_a9_0 
	@$(ECHO) "Quartus flow complete"


copy:
	rm -rf output
	mkdir output
	cp -f *.rbf output/
	cp -f hps_0.h output/
	cp -f HLS/csynth.prj/components/ACCL_TOP/ACCL_TOP_csr.h output/
	cp -f DE10_NANO.fit.summary output/
	cp -f DE10_NANO.sta.summary output/
	cp -f output/* ../Host/ 

clean:
	rm -f *.txt *.rpt *.smsg *.summary *.jdi *.pin *.sld *.csv 
	rm -f *.done *.sof *.sopcinfo *.qws soc_system.BAK.qsys *.h *.rbf
	rm -rf db incremental_db hps_isw_handoff .qsys_edit soc_system
