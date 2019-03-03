#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

class Gemm{
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
    cout << "size of B = " << B.size() << "x" << B[0].size() << endl;
    cout << "size of C = " << C.size() << endl;
    cout << "input size = " << B[0].size() << endl;
    cout << "output size = " << C.size() << endl << endl;
}

vector<double> relu(const vector<double> &x){
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

int main(){
    MNIST mnist("mnist_test.txt");
    mnist.show();

    Gemm g1("140406444019384_matrix.txt", "140406130172200_matrix.txt");
    Gemm g2("140406443536680_matrix.txt", "140406443536904_matrix.txt");
    Gemm g3("140406443537240_matrix.txt", "140406443537464_matrix.txt");
    g1.show();
    g2.show();
    g3.show();
   
    // int n = mnist.answer.size(); 
    vector<int> o;
    for(int i=0;i<1000;i++){
        vector<double> x = mnist.input[i];
        x = g1.calc(x);
        x = relu(x);
        x = g2.calc(x);
        x = relu(x);
        x = g3.calc(x);
        int a = distance(x.begin(), max_element(x.begin(), x.end()));
        o.push_back(a);
    }
    double ac = mnist.accuracy(o);
    cout << "Accuracy = " << ac << endl;
    return 0;
}
