
/* This header file describes the CSR Slave for the ACCL_TOP component */

#ifndef __ACCL_TOP_CSR_REGS_H__
#define __ACCL_TOP_CSR_REGS_H__



/******************************************************************************/
/* Memory Map Summary                                                         */
/******************************************************************************/

/*
  Register  | Access  |   Register Contents      | Description
  Address   |         |      (64-bits)           | 
------------|---------|--------------------------|-----------------------------
        0x0 |       R |         {reserved[62:0], |     Read the busy status of
            |         |               busy[0:0]} |               the component
            |         |                          |  0 - the component is ready
            |         |                          |       to accept a new start
            |         |                          |    1 - the component cannot
            |         |                          |          accept a new start
------------|---------|--------------------------|-----------------------------
        0x8 |       W |         {reserved[62:0], |  Write 1 to signal start to
            |         |              start[0:0]} |               the component
------------|---------|--------------------------|-----------------------------
       0x10 |     R/W |         {reserved[62:0], |      0 - Disable interrupt,
            |         |   interrupt_enable[0:0]} |        1 - Enable interrupt
------------|---------|--------------------------|-----------------------------
       0x18 |  R/Wclr |         {reserved[61:0], | Signals component completion
            |         |               done[0:0], |       done is read-only and
            |         |   interrupt_status[0:0]} | interrupt_status is write 1
            |         |                          |                    to clear
------------|---------|--------------------------|-----------------------------
       0x20 |     R/W |     {input_buffer[63:0]} |       Argument input_buffer
------------|---------|--------------------------|-----------------------------
       0x28 |     R/W |    {output_buffer[63:0]} |      Argument output_buffer
------------|---------|--------------------------|-----------------------------
       0x30 |     R/W |         {reserved[31:0], |            Argument seq_len
            |         |           seq_len[31:0]} |                            

NOTE: Writes to reserved bits will be ignored and reads from reserved
      bits will return undefined values.
*/


/******************************************************************************/
/* Register Address Macros                                                    */
/******************************************************************************/

/* Byte Addresses */
#define ACCL_TOP_CSR_BUSY_REG (0x0)
#define ACCL_TOP_CSR_START_REG (0x8)
#define ACCL_TOP_CSR_INTERRUPT_ENABLE_REG (0x10)
#define ACCL_TOP_CSR_INTERRUPT_STATUS_REG (0x18)
#define ACCL_TOP_CSR_ARG_INPUT_BUFFER_REG (0x20)
#define ACCL_TOP_CSR_ARG_OUTPUT_BUFFER_REG (0x28)
#define ACCL_TOP_CSR_ARG_SEQ_LEN_REG (0x30)

/* Argument Sizes (bytes) */
#define ACCL_TOP_CSR_ARG_INPUT_BUFFER_SIZE (8)
#define ACCL_TOP_CSR_ARG_OUTPUT_BUFFER_SIZE (8)
#define ACCL_TOP_CSR_ARG_SEQ_LEN_SIZE (4)

/* Argument Masks */
#define ACCL_TOP_CSR_ARG_INPUT_BUFFER_MASK (0xffffffffffffffffULL)
#define ACCL_TOP_CSR_ARG_OUTPUT_BUFFER_MASK (0xffffffffffffffffULL)
#define ACCL_TOP_CSR_ARG_SEQ_LEN_MASK (0xffffffff)

/* Status/Control Masks */
#define ACCL_TOP_CSR_BUSY_MASK   (1<<0)
#define ACCL_TOP_CSR_BUSY_OFFSET (0)

#define ACCL_TOP_CSR_START_MASK   (1<<0)
#define ACCL_TOP_CSR_START_OFFSET (0)

#define ACCL_TOP_CSR_INTERRUPT_ENABLE_MASK   (1<<0)
#define ACCL_TOP_CSR_INTERRUPT_ENABLE_OFFSET (0)

#define ACCL_TOP_CSR_INTERRUPT_STATUS_MASK   (1<<0)
#define ACCL_TOP_CSR_INTERRUPT_STATUS_OFFSET (0)
#define ACCL_TOP_CSR_DONE_MASK   (1<<1)
#define ACCL_TOP_CSR_DONE_OFFSET (1)


#endif /* __ACCL_TOP_CSR_REGS_H__ */



