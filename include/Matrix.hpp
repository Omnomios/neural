#ifndef matrix_hpp
#define matrix_hpp

#include <random>
#include <vector>

class Matrix
{
public:
    Matrix();
    Matrix(int row, int col): data(row, std::vector<double>(col, 0)) {};
    ~Matrix();

    void set(std::vector<std::vector<double>> data) 
    {
        this->data = data;
    }
    
    Matrix& random(double low, double high)
    {
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(low, high);
        return *this;
    }

    size_t rows() { return this->data.size(); };
    size_t cols() { return this->data[0].size(); };

    Matrix col(int n) {
        return Matrix(this->cols(), 1);
    };
    Matrix row(int n) {
        return Matrix(this->rows(), 1);
    };

    Matrix operator+ (Matrix const& b) {
        return Matrix(this->rows(), this->cols());
    }

    Matrix& operator+=(const Matrix& rhs)
    {
        //TODO: RHS addition
        return *this;
    }

    Matrix& operator+=(const double& rhs)
    {
        //TODO: RHS addition
        return *this;
    }

    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;

private:
    std::vector<std::vector<double>> data;
};


double& Matrix::operator()(int row, int col)
{
    return this->data[row][col];
}
 
const double& Matrix::operator()(int row, int col) const
{
    return this->data[row][col];
}

#endif