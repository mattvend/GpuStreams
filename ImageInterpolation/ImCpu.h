/**********************************************************************************/
/* The MIT License(MIT)                                                           */
/*                                                                                */
/* Copyright(c) 2016-2016 Matthieu Vendeville                                     */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions :                      */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE     */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#pragma once

#include "Im.h"

class ImCpu: public Im
{
public:
	ImCpu(const char* filename);
    ImCpu(const ImCpu &); // Copy constructor
	~ImCpu(void);

	virtual ImCpu* clone() const { return new ImCpu(*this); };
	
	void InterpolateNN(unsigned short new_width, unsigned short new_height);
	void InterpolateBilinear(unsigned short new_width, unsigned short new_height);
	void Save2RawFile(const char* filename);
	void PrintRawFileName();

private:
	unsigned short width;       /* Image Width in pixels*/
	unsigned short height;      /* Image Height in pixels*/
	unsigned short bpp;         /* Bits per Pixel. Possible values are 8 or 16 */
	unsigned short dimension;   /* Dimension of the image, or numbers of channels*/
	void*   pxl;                /* Actual pixels stored in a row array */
	char    raw_file_name[256]; /* raw file full file name when saved on the disk */
};

