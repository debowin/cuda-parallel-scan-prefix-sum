#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  for( unsigned int i = 1; i < len; ++i) 
  {
      total_sum += idata[i-1];
      reference[i] = idata[i-1] + reference[i-1];
  }
  if (total_sum != reference[len-1])
      printf("Warning: exceeding single-precision accuracy.  Scan will have reduced precision.\n");
  
}