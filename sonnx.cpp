#include <vector>
#include <iomanip>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

class Node{
    public:
        virtual std::vector<double> calc(const std::vector<double> &x) = 0;
};

class CompressedGemm : Node{
    public:
        CompressedGemm(const std::string &b_name, const std::string &c_name, const double compress_ratio);
        std::vector<double> calc(const std::vector<double> &x);
        void show();
        constexpr static const double DEFAULT_COMPRESS_RATIO = 0.8;
    private:
        std::vector<std::vector<double> > B;
        std::vector<double> C;
};

CompressedGemm::CompressedGemm(const std::string &b_name, const std::string &c_name, const double compress_ratio = CompressedGemm::DEFAULT_COMPRESS_RATIO){
    std::ifstream bf(b_name);
    std::string s;
    vector<double> as;
    while(getline(bf, s)){
        std::stringstream ss(s);
        double d;
        vector<double> vd;
        while(ss>>d){
            vd.push_back(d);
            as.push_back(abs(d));
        }
        this->B.push_back(vd);
    }

    sort(as.begin(), as.end());
    double th = as[(int)(as.size() * compress_ratio)];
    for(int i=0; i < B.size(); i++){
        for(int j=0;j < B[i].size(); j++){
            if(abs(B[i][j]) < th){
                B[i][j] = 0;
            }
        }
    }

    ifstream cf(c_name);
    double d;
    vector<double> vd;
    while(cf>>d){
        vd.push_back(d);
    }
    this->C = vd;
}

vector<double> CompressedGemm::calc(const vector<double> &x){
    vector<double> ret = C;
    int n = this->B.size();
    int m = this->B[0].size();
    for(int i=0; i < n; i++){
        for(int j=0; j< m;j++){
            ret[i] += B[i][j] * x[j];
        }
    }
    return ret;
}

void CompressedGemm::show(){
    cout << "CompressedGemm" << endl;
    cout << "size of B = " << B.size() << "x" << B[0].size() << endl;
    cout << "size of C = " << C.size() << endl;
    cout << "input size = " << B[0].size() << endl;
    cout << "output size = " << C.size() << endl << endl;
}

class Gemm : Node{
    public:
        Gemm(const std::string &b_name, const std::string &c_name);
        std::vector<double> calc(const std::vector<double> &x);
        void show();
    private:
        std::vector<std::vector<double> > B;
        std::vector<double> C;
};

Gemm::Gemm(const std::string &b_name, const std::string &c_name){
    std::ifstream bf(b_name);
    std::string s;
    while(getline(bf, s)){
        std::stringstream ss(s);
        double d;
        vector<double> vd;
        while(ss>>d){
            vd.push_back(d);
        }
        this->B.push_back(vd);
    }
    ifstream cf(c_name);
    double d;
    vector<double> vd;
    while(cf>>d){
        vd.push_back(d);
    }
    this->C = vd;
}

vector<double> Gemm::calc(const vector<double> &x){
    vector<double> ret = C;
    int n = this->B.size();
    int m = this->B[0].size();
    for(int i=0; i < n; i++){
        for(int j=0; j< m;j++){
            ret[i] += B[i][j] * x[j];
        }
    }
    return ret;
}

void Gemm::show(){
    cout << "Gemm" << endl;
    cout << "size of B = " << B.size() << "x" << B[0].size() << endl;
    cout << "size of C = " << C.size() << endl;
    cout << "input size = " << B[0].size() << endl;
    cout << "output size = " << C.size() << endl << endl;
}

class Relu : Node{
    public:
        std::vector<double> calc(const std::vector<double> &x);
};

vector<double> Relu::calc(const vector<double> &x){
    vector<double> res = x;
    int n = res.size();
    for(int i=0;i<n;i++){
        res[i] = max(0.0,res[i]);
    }
    return res;
}

class MNIST{
    public:
        static const int MNIST_SIZE = 784;
        vector<vector<double>> input;
        vector<int> answer;
        MNIST(const string &f);
        const void show();
        const double accuracy(const vector<int> &x);
};

MNIST::MNIST(const string &f){
    std::ifstream bf(f);
    std::string s;
    while(getline(bf, s)){
        std::stringstream ss(s);
        double d;
        vector<double> vd;
        for(int i=0;i<this->MNIST_SIZE;i++){
            ss>>d;
            vd.push_back(d);
        }
        int a;
        ss>>a;
        this->input.push_back(vd);
        this->answer.push_back(a);
    }

}

const void MNIST::show(){
    cout<<"size of MNIST = " << this->input.size() << endl << endl;
}

const double MNIST::accuracy(const vector<int> &x){
    int n = x.size();
    int c = 0;
    for(int i=0;i<n;i++){
        if(x[i] == this->answer[i]){
            c++;
        }
    }
    return (double)c/(double)n;
}

template<typename T>
const void show_vector(const vector<T> &v){
    for(auto x:v){
        cout<<x<<" ";
    }
    cout<<endl;
}

double original_graph_accuracy(MNIST &mnist, const int ntest){
    Gemm g1("140406444019384_matrix.txt", "140406130172200_matrix.txt");
    Gemm g2("140406443536680_matrix.txt", "140406443536904_matrix.txt");
    Gemm g3("140406443537240_matrix.txt", "140406443537464_matrix.txt");
    Relu r;

    vector<int> o;
    for(int i=0;i<ntest ;i++){
        vector<double> x = mnist.input[i];
        x = g1.calc(x);
        x = r.calc(x);
        x = g2.calc(x);
        x = r.calc(x);
        x = g3.calc(x);
        int a = distance(x.begin(), max_element(x.begin(), x.end()));
        o.push_back(a);
    }
    double ac = mnist.accuracy(o);
    return ac;
}

double compressed_graph_accuracy(MNIST &mnist, const int ntest, double compress_ratio){
    CompressedGemm g1("140406444019384_matrix.txt", "140406130172200_matrix.txt", compress_ratio);
    CompressedGemm g2("140406443536680_matrix.txt", "140406443536904_matrix.txt", compress_ratio);
    CompressedGemm g3("140406443537240_matrix.txt", "140406443537464_matrix.txt", compress_ratio);
    Relu r;

    vector<int> o;
    for(int i=0;i<ntest ;i++){
        vector<double> x = mnist.input[i];
        x = g1.calc(x);
        x = r.calc(x);
        x = g2.calc(x);
        x = r.calc(x);
        x = g3.calc(x);
        int a = distance(x.begin(), max_element(x.begin(), x.end()));
        o.push_back(a);
    }
    double ac = mnist.accuracy(o);
    return ac;
}

int main(){
    MNIST mnist("mnist_test.txt");

    int n = mnist.answer.size(); 
    cout << "compress_ratio, accuracy" << endl;
    for(int r = 0; r < 80; r+=5){
        double ac = compressed_graph_accuracy(mnist, n, (double)r/100.0);
        cout << setprecision(10) << (double)r/100.0 << ", " << ac << endl;
    }
    for(int r = 80; r <= 100; r+=1){
        double ac = compressed_graph_accuracy(mnist, n, (double)r/100.0);
        cout << setprecision(10) << (double)r/100.0 << ", " << ac << endl;
    }
    return 0;
}
