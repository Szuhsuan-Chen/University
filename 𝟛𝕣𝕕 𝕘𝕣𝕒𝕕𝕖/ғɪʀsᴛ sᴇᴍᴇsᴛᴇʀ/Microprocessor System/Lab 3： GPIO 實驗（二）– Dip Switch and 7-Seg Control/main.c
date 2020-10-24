#include <nds32_intrinsic.h>
#include "WT58F2C9.h"
#include "gpio.h"


void delay1(unsigned int nCount)
{
   volatile unsigned int i;
   for (i=0;i<nCount;i++);
}


/* Intrrupt memory address */
#define INT_MEM_ADDR_BASE			(0x00200D00)
#define rINT0_IE0_ENABLE			(INT_MEM_ADDR_BASE+0x00)
#define rINT0_IE1_ENABLE			(INT_MEM_ADDR_BASE+0x04)
#define rINT0_IE2_ENABLE			(INT_MEM_ADDR_BASE+0x08)
#define rINT1_IE0_ENABLE			(INT_MEM_ADDR_BASE+0x10)
#define rINT1_IE1_ENABLE			(INT_MEM_ADDR_BASE+0x14)
#define rINT1_IE2_ENABLE			(INT_MEM_ADDR_BASE+0x18)

#define rINT0_IE0_FLAG				(INT_MEM_ADDR_BASE+0x20)
#define rINT0_IE1_FLAG				(INT_MEM_ADDR_BASE+0x24)
#define rINT0_IE2_FLAG				(INT_MEM_ADDR_BASE+0x28)
#define rINT1_IE0_FLAG				(INT_MEM_ADDR_BASE+0x30)
#define rINT1_IE1_FLAG				(INT_MEM_ADDR_BASE+0x34)
#define rINT1_IE2_FLAG				(INT_MEM_ADDR_BASE+0x38)


inline void GIE_ENABLE();

void DRV_EnableHWInt (void)
{
	/* enable SW0, HW0 and HW1 */
	__nds32__mtsr(0x10003, NDS32_SR_INT_MASK);
	/* Enable SW0 */
	//__nds32__mtsr(0x10000, NDS32_SR_INT_MASK);
	/* Enable global interrupt */
	GIE_ENABLE();
}

void DRV_BlockIntDisable(void)
{
	// Disable all interrupt
	OUTW(rINT0_IE0_ENABLE, 0x0000);
	OUTW(rINT0_IE1_ENABLE, 0x0000);
	OUTW(rINT0_IE2_ENABLE, 0x0000);
	OUTW(rINT1_IE0_ENABLE, 0x0000);
	OUTW(rINT1_IE1_ENABLE, 0x0000);
	OUTW(rINT1_IE2_ENABLE, 0x0000);
}

void DRV_IntInitial(void)
{
	// Disable all interrupt
	DRV_BlockIntDisable();

	// Enable all HW interrupt
	DRV_EnableHWInt();				//Enable global Hardware interrupt, Assembly command

	// Enable default Block interrupt
	//DRV_BlockIntEnable();			//Enable each block device interrupt

}

#define rSYS_OPTION1	(0x00200004)

void DRV_SysXtal(U8 u8XtalMode)
{

	#if(EXTERNAL_XTAL == XTAL_MODE)
		//-----External Crystal
		//-----24MHz
		OUTW(rSYS_OPTION1,((INW(rSYS_OPTION1)&0xFFFFFF00) | 0x0012)); //Use HXTAL and divide 2
		//-----Crystal 12MHz
		//OUTW(rSYS_OPTION1, (INW(rSYS_OPTION1) | 0x000A));

		//OUTW(rSYS_OPTION1,(INW(rSYS_OPTION1) | (XTAL<<2) | (HSE_OSC_ON <<1)));
		//OUTW(rSYS_CLOCK_SELECT,(INW(rSYS_CLOCK_SELECT) | 0x0001));
	#else
		//-----Internal RC
		//-----24MHz
		//OUTW(rSYS_OPTION1,(INW(rSYS_OPTION1) | 0x0012));
		//-----Crystal 12MHz
		//OUTW(rSYS_OPTION1, (INW(rSYS_OPTION1) | 0x000A));

		//OUTW(rSYS_OPTION1,(INW(rSYS_OPTION1) | (XTAL<<2)));
		//OUTW(rSYS_CLOCK_SELECT,(INW(rSYS_CLOCK_SELECT));
	#endif
	//-----MCU Clock Output Test
	//OUTW(rSYS_OPTION3,0x0090);
}

void OS_PowerOnDriverInitial(void)
{
	//========================= << Typedef Initial  >>
	//SYS_TypeDefInitial();
	//========================= << System Clock Initial >>
	//-----External Crystal
	DRV_SysXtal(EXTERNAL_XTAL);
	//========================= << Interrupt Initial >>
	DRV_IntInitial();
	//========================= << GPIO Initial >>
	//DRV_GpioInitial();
	//========================= << UART Initial >>
	//DRV_UartInitial();
	//========================= << Timer Initial >>
	//DRV_TimerInitial(TIMER_0, SIMPLE_TIMER);
	//DRV_TimerInitial(TIMER_1, SIMPLE_TIMER);
	//========================= << PWM Initial >>
	//DRV_PwmInitial();
	//========================= << SPI Initial >>
	//DRV_SpiInitial(SPI_CH2);
	//========================= << Watchdog Initial >>
}




//-----  UART Functions  -----
#define UART3_ADDR_BASE		(0x0020B400)
#define UART3_CR1			(UART3_ADDR_BASE+0x00)
#define UART3_CR2			(UART3_ADDR_BASE+0x04)

//0:1 Start bit/8 Data bits/1 Stop bit,
#define	UART_FORMAT_N81	0
//1:1 Start bit/9 Data bits/1 Stop bit
#define	UART_FORMAT_N91	1

#define UART_EN				(1<<17)	//0:Disable, 1:Enable
#define UART_TX_EN			(1<<15)
#define UART_RX_EN			(1<<14)
#define UART_OVER8			(1<<13)
#define UART_OVER16			(0<<13)
#define UART_WORD_LENGTH	(UART_FORMAT_N81<<12)//	//0:1 Start bit/8 Data bits/1 Stop bit, //1:1 Start bit/9 Data bits/1 Stop bit
#define UART_TX_DMA_EN		(1<<11) //0:Disable, 1:Enable
#define UART_RX_DMA_EN		(1<<10)	// 0:Disable, 1:Enable
#define UART_RX_WAKEUP		(0<<9)	// 0:In active mode, 1:In mute mode
#define UART_WAKEUP_METHOD	(0<<8)	// 0:Idle mode, 1:Address mark
#define UART_UART_ADDR_NODE	(0x5<<4)	//Data 0x0=xxxx0000b ~ 0xF=xxxx1111b,
#define UART_PARITY_EN		(0<<2)//SET_BIT1//(n<<1) //0:Disable, 1:Enable
#define UART_PARITY_SEL		(1<<1)// O:Even 1:Odd
#define UART_STOP_BIT	0// O:1-bit 1:2-bit

#define UART_SET_CTL_PARA	UART_EN|UART_TX_EN|UART_RX_EN|UART_WORD_LENGTH\
							|UART_RX_WAKEUP|UART_WAKEUP_METHOD|UART_UART_ADDR_NODE\
							|UART_PARITY_EN|UART_PARITY_SEL|UART_STOP_BIT

//HXTAL = HXTAL_24M (OVER8=0)
//9600 = 156.25
#define BUARRATE_9600_MANTISSA_24MHZ	156
#define BUARRATE_9600_FRACTION_24MHZ	4
//19200 = 78.125
#define BUARRATE_19200_MANTISSA_24MHZ	78
#define BUARRATE_19200_FRACTION_24MHZ	2
//38400 = 39.0625
#define BUARRATE_38400_MANTISSA_24MHZ	39
#define BUARRATE_38400_FRACTION_24MHZ	1
//57600 = 26.0625
#define BUARRATE_57600_MANTISSA_24MHZ	26
#define BUARRATE_57600_FRACTION_24MHZ	1
//115200 = 13.0
#define BUARRATE_115200_MANTISSA_24MHZ	13
#define BUARRATE_115200_FRACTION_24MHZ	0
//230400 = 6.5
#define BUARRATE_230400_MANTISSA_24MHZ	6
#define BUARRATE_230400_FRACTION_24MHZ	8
//921600 = 1.625
#define BUARRATE_921600_MANTISSA_24MHZ	1
#define BUARRATE_921600_FRACTION_24MHZ	10


#define FALSE 0
#define TRUE  1
#define UART_TXD_BUFFER_SIZE	24
#define _EOS_	'\0' //End of string

char u8TxdBuf[UART_TXD_BUFFER_SIZE];

void DRV_PutChar(char u8Char)
{
	U16 u16Count;

	OUTW(UART3_ADDR_BASE+0x0C, u8Char);

	u16Count = 0;
	//Wait transmission complete then clear by SW write to 0
	while((INW(UART3_ADDR_BASE+0x08)&0x00000020) == 0)
	{
	#if 1 //Don't delete.
		//-----Time out
		u16Count++;
		if(u16Count >= 1000)
		{
			break;
		}
	#endif
	}
	OUTW(UART3_ADDR_BASE+0x08, INW(UART3_ADDR_BASE+0x08) & 0xFFFFFFDF);
}

void DRV_PutStr(const char *pFmt)
{
	U8 u8Buffer;	//Character buffer

	while (1)
	{
		u8Buffer = *pFmt; //Get a character
		if(u8Buffer == _EOS_) //Check end of string
			break;

		DRV_PutChar(u8Buffer); //Put a character
		pFmt++;
	}
}

void DRV_IntToStr(U16 u16Val, U8 u8Base, char *pBuf, U8 u8Length)
{
	U8 bShowZero = FALSE;
	U16 u16Divider;
	U8 u8Disp;
	U16 u16Temp;

	u8Length -= 1;
	if(u8Base == 16) //Hex
	{
		u16Temp = 0x01 << u8Length;
	}
	else //Dec
	{
		u16Temp = 1;
		while(u8Length--)
		{
			u16Temp *= 10;
		}
	}

	if( 0 == u16Val )
	{
		if( 16 == u8Base )
		{
			pBuf[0] = '0';
			pBuf[1] = '0';
			pBuf[2] = '0';
			pBuf[3] = '0';
			pBuf[4] = '\0';
		}
		else
		{
			pBuf[0] = '0';
			pBuf[1] = '0';
			pBuf[2] = '0';
			pBuf[3] = '0';
			pBuf[4] = '0';
			pBuf[5] = '\0';
		}
		return;
	}

	if( 16 == u8Base )
	{
		u16Divider = 0x1000;
	}
	else
	{
		u16Divider = 10000;
	}

	while( u16Divider )
	{
		u8Disp = u16Val / u16Divider;
		u16Val = u16Val % u16Divider;

		if(u16Temp == u16Divider)
		{
			bShowZero = TRUE;
		}
		if( u8Disp || bShowZero || (u16Divider>0))
		{
			if( u8Disp < 10 )
				*pBuf = '0' + u8Disp;
			else
				*pBuf = 'A' + u8Disp - 10;
			pBuf ++;
		}

		if( 16 == u8Base )
		{
			u16Divider /= 0x10;
			if(bShowZero)
				u16Temp /= 0x10;
		}
		else
		{
			u16Divider /= 10;
			if(bShowZero)
				u16Temp /= 10;
		}
	}
	*pBuf = '\0';
}

void DRV_Printf(char *pFmt, U16 u16Val)
{
	U8 u8Buffer;

	//-----Pin configuration for UART3
	GPIO_PTC_FS = 0x0300;
	GPIO_PTC_PADINSEL = 0x0000;
	GPIO_PTC_DIR = 0xFEFF;
	GPIO_PTC_CFG = 0x0000;

	//UART Parameter
	OUTW(UART3_ADDR_BASE+0x00, UART_SET_CTL_PARA);

	//Set Baud rate with default sysclk
	OUTW(UART3_ADDR_BASE+0x14, ((BUARRATE_38400_MANTISSA_24MHZ<<4)|BUARRATE_38400_FRACTION_24MHZ)/2); //38400 for 12MHz MCUCLK

	while((u8Buffer =(U8)*(pFmt++)))
	{
		if(u8Buffer == '%') //check special case
		{
			switch(*(pFmt++)) //check next character
			{
				case 'x': //hexadecimal number
				case 'X':
					DRV_IntToStr(u16Val, 16, u8TxdBuf, 2);
					DRV_PutStr(u8TxdBuf);
				break;
				case 'd': //decimal number
				case 'i':
					DRV_IntToStr(u16Val, 10, u8TxdBuf,5);
					DRV_PutStr(u8TxdBuf);
				break;
				case 'c':
				case 'C':
					DRV_PutChar((char)u16Val);
				break;
			} //switch
		}
		else //general
		{
			DRV_PutChar(u8Buffer); //put a character
		}
	}
}



int main()
{
	unsigned int i, myi, k, tmp, j;
	//Digit_8 Digit_7 Digit_6 Digit_5 Digit_4 Digit_3 Digit_2 Digit_1
	unsigned int index_7LED[8] = {Digit_1, Digit_2, Digit_3, Digit_4, Digit_5, Digit_6, Digit_7, Digit_8}; //顯示的位置
	unsigned int index_7LED_NUM[8] = {Number_4, Number_0, Number_7, Number_2, Number_6, Number_1, Number_4, Number_1}; //顯示的數字


	OS_PowerOnDriverInitial();


	DRV_Printf("====================================\r\n", 0);
	DRV_Printf("   ADP-WT58F2C9 7-SEG demo program   \r\n", 0);
	DRV_Printf("====================================\r\n", 0);


	DRV_Printf("7-SEG testing ...\r\n", 0);

	// Setting for 7LED select
	GPIO_PTA_DIR = 0x0000;
	GPIO_PTA_CFG = 0x0000;
	GPIO_PTA_GPIO = Digit_1;
	// Setting for 7LED number
	GPIO_PTD_DIR = 0x0000;
	GPIO_PTD_CFG = 0x0000;
	//GPIO_PTD_GPIO = 0xFFFF;

	
	while(1){
		//讀取Dip_Switch的值
		tmp = (GPIO_PTC_PADIN >> 2) & 0x0000000F;
		//Dip_Switch 1
		//由左至右飛入(最末位數字先出現)
		if(tmp == 1){
			for(i=0; i<8; i++){ //一次多出現一個數字
				for (myi=0; myi<2000; myi++){ //因為印數字出來不是同時，所以讓它跑很多次迴圈會有視覺暫留的效果
					int temp = 7-i;
					for(k=0; k<=i; k++){ //把同一時間該出現的數字印出來
						GPIO_PTA_GPIO = index_7LED[7-k]; //7-0
						GPIO_PTD_GPIO = index_7LED_NUM[temp]; 
						temp++;
						delay1(20); //避免全部糊在一起
					}
				}
			}
			//持續顯示所有的學號
			while(1){
				//當偵測到Dip_Switch不為1號時，就跳出迴圈
				if(((GPIO_PTC_PADIN >> 2) & 0x0000000F) != 1){
					break;
				}

				for (myi=0; myi<2000; myi++){ //視覺暫留
					int temp = i;
					for(k=0; k<=7; k++){ //把8個位置上對應的數字都印出來
						GPIO_PTA_GPIO = index_7LED[k];
						GPIO_PTD_GPIO = index_7LED_NUM[7-k];
						temp--;
						delay1(20); //避免全部糊在一起
					}
				}

			}
			//清空 全滅
			GPIO_PTD_GPIO = 0x0000;
		}
		//Dip_Switch 2
		//由右至左飛入(第一位數字先出現)
		if(tmp == 2){
			for(i=0; i<8; i++){ //一次多出現一個數字
				for (myi=0; myi<2000; myi++){ //因為印數字出來不是同時，所以讓它跑很多次迴圈會有視覺暫留的效果
					int temp = i;
					for(k=0; k<=i; k++){ //把同一時間該出現的數字印出來
						GPIO_PTA_GPIO = index_7LED[k]; //0-7
						GPIO_PTD_GPIO = index_7LED_NUM[temp];
						temp--;
						delay1(20); //避免全部糊在一起
					}
				}
			}
			//持續顯示所有的學號
			while(1){
				//當偵測到Dip_Switch不為2號時，就跳出迴圈
				if(((GPIO_PTC_PADIN >> 2) & 0x0000000F) != 2){
					break;
				}

				for (myi=0; myi<2000; myi++){ //視覺暫留
				int temp = i;
					for(k=0; k<=7; k++){ //把8個位置上對應的數字都印出來
						GPIO_PTA_GPIO = index_7LED[k];
						GPIO_PTD_GPIO = index_7LED_NUM[7-k];
						temp--;
						delay1(20); //避免全部糊在一起
					}
				}

			}
			//清空 全滅
			GPIO_PTD_GPIO = 0x0000;
		}
		//Dip_Switch 3
		//由左至右飛入7-SEG ，最末位數字先出現在Digit_8，右移到定位(Digit_1)時，停住顯示不動，其他數字依序飛入，飛到定位時，停住顯示不動
		if(tmp == 4){
			for(i=0; i<8; i++){
				for (j=0; j<=7-i; j++){//同一數字右移
					for(myi=0; myi<2000; myi++){
						//已堆積在右邊的數字
						for(k=0; k<i; k++){
							GPIO_PTA_GPIO = index_7LED[k];
							GPIO_PTD_GPIO = index_7LED_NUM[7-k];
							delay1(20);
						}
						//同一數字右移
						GPIO_PTA_GPIO = index_7LED[7-j];
						GPIO_PTD_GPIO = index_7LED_NUM[7-i];
						delay1(20);
						GPIO_PTD_GPIO = 0x0000;
					}
				}
			}
			//持續顯示所有的學號
			while(1){
				//當偵測到Dip_Switch不為3號時，就跳出迴圈
				if(((GPIO_PTC_PADIN >> 2) & 0x0000000F) != 4){
					break;
				}

				for (myi=0; myi<2000; myi++){ //視覺暫留
				int temp = i;
					for(k=0; k<=7; k++){ //把8個位置上對應的數字都印出來
						GPIO_PTA_GPIO = index_7LED[k];
						GPIO_PTD_GPIO = index_7LED_NUM[7-k];
						temp--;
						delay1(20); //避免全部糊在一起
					}
				}

			}
			//清空 全滅
			GPIO_PTD_GPIO = 0x0000;
		}
		//Dip_Switch 4
		//由右至左飛入7-SEG ，第一位數字先出現在Digit_1，左移到定位(Digit_8)時，停住顯示不動，其他數字依序飛入，飛到定位時，停住不動顯示
		if(tmp == 8){
			for(i=0; i<8; i++){
				for (j=0; j<=7-i; j++){ ////同一數字右移
					for(myi=0; myi<2000; myi++){
						//已堆積在右邊的數字
						for(k=0; k<i; k++){
							GPIO_PTA_GPIO = index_7LED[7-k];
							GPIO_PTD_GPIO = index_7LED_NUM[k];
							delay1(20);
						}
						//同一數字右移
						GPIO_PTA_GPIO = index_7LED[j];
						GPIO_PTD_GPIO = index_7LED_NUM[i];
						delay1(20);
						GPIO_PTD_GPIO = 0x0000;
					}
				}
			}
			//持續顯示所有的學號
			while(1){
				//當偵測到Dip_Switch不為4號時，就跳出迴圈
				if(((GPIO_PTC_PADIN >> 2) & 0x0000000F) != 8){
					break;
				}

					for (myi=0; myi<2000; myi++){ //視覺暫留
					int temp = i;
						for(k=0; k<=7; k++){ //把8個位置上對應的數字都印出來
							GPIO_PTA_GPIO = index_7LED[k];
							GPIO_PTD_GPIO = index_7LED_NUM[7-k];
							temp--;
							delay1(20); //避免全部糊在一起
						}
					}

			}
			//清空 全滅
			GPIO_PTD_GPIO = 0x0000;
		}



	}








	DRV_Printf("====================================\r\n", 0);


	return 0;
}





