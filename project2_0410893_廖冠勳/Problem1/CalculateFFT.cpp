#include <complex>
#include <iostream>
#include <valarray>
#include <fstream>
#include <vector>

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft( x );

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
}

int main(int argc, char const *argv[])
{

    std::ifstream infile;
    std::ofstream outfile;
    infile.open("problem1.txt");
    outfile.open("ans.txt");
    double realnum,imaginum;
    int amtElement=0;
    std::vector<Complex> test;
    while(infile>>realnum>>imaginum)
    {
      Complex temp(realnum,imaginum);
      test.push_back(temp);
      amtElement++;
    }
    Complex* comparray = &test[0]; // convert a vector to array
    CArray data(comparray,amtElement);


    // forward fft
    fft(data);

    outfile << "fft" << std::endl;
    for (int i = 0; i < amtElement; ++i)
    {
        if(data[i].imag()<0)outfile << data[i].real()<<data[i].imag()<<"i"<< std::endl;
        else outfile << data[i].real()<<"+"<<data[i].imag()<<"i"<< std::endl;
    }

    // inverse fft
    ifft(data);

    outfile << std::endl << "ifft" << std::endl;
    for (int i = 0; i < amtElement; ++i)
    {
        if(data[i].imag()<0)
        {
          if(data[i].real()<0.00001 ) outfile <<data[i].imag()<<"i"<< std::endl;
          else if(data[i].imag()>-0.00001 && data[i].imag()<1) outfile << data[i].real()<< std::endl;
          else outfile << data[i].real()<<data[i].imag()<<"i"<< std::endl;
        }
        else
        {
          if(data[i].real()<0.0001 ) outfile <<data[i].imag()<<"i"<< std::endl;
          else if(data[i].imag()<0.0001) outfile << data[i].real()<< std::endl;
          else outfile << data[i].real()<<"+"<<data[i].imag()<<"i"<< std::endl;

        }
    }

    infile.close();
    outfile.close();
    return 0;
}
