#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

class Node{
    public:
        virtual std::vector<float> calc(const std::vector<float> &x) = 0;
};

class CompressedGemm : Node{
    public:
        CompressedGemm(const std::string &b_name, const std::string &c_name, const float compress_ratio);
        std::vector<float> calc(const std::vector<float> &x);
        void show();
        constexpr static const float DEFAULT_COMPRESS_RATIO = 0.8;
    private:
        std::vector<std::vector<float> > B;
        std::vector<float> C;
        vector<float> B_scale;
        vector<int> B_row;
        vector<int> B_column;
};

CompressedGemm::CompressedGemm(const std::string &b_name, const std::string &c_name, const float compress_ratio = CompressedGemm::DEFAULT_COMPRESS_RATIO){
    std::ifstream bf(b_name);
    std::string s;
    vector<float> as;
    while(getline(bf, s)){
        std::stringstream ss(s);
        float d;
        vector<float> vd;
        while(ss>>d){
            vd.push_back(d);
            as.push_back(abs(d));
        }
        this->B.push_back(vd);
    }

    sort(as.begin(), as.end());
    float th = as[(int)(as.size() * compress_ratio)];
    for(int i=0; i < B.size(); i++){
        for(int j=0;j < B[i].size(); j++){
            if(abs(B[i][j]) < th){
                B[i][j] = 0;
            }else{
                B_scale.push_back(B[i][j]);
                B_row.push_back(i);
                B_column.push_back(j);
            }
        }
    }

    ifstream cf(c_name);
    float d;
    vector<float> vd;
    while(cf>>d){
        vd.push_back(d);
    }
    this->C = vd;
}

vector<float> CompressedGemm::calc(const vector<float> &x){
    vector<float> ret = C;
    // int n = B.size();
    // int m = B[0].size();
    // for(int i=0; i < n; i++){
    //     for(int j=0; j< m;j++){
    //         ret[i] += B[i][j] * x[j];
    //     }
    // }

    int n = B_scale.size();
    for(int i=0;i+3<n;i+=4){
        ret[B_row[i]] += B_scale[i] * x[B_column[i]];
        ret[B_row[i+1]] += B_scale[i+1] * x[B_column[i+1]];
        ret[B_row[i+2]] += B_scale[i+2] * x[B_column[i+2]];
        ret[B_row[i+3]] += B_scale[i+3] * x[B_column[i+3]];
    }
    for(int i=n/4*4;i<n;i++){
        ret[B_row[i]] += B_scale[i] * x[B_column[i]];
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
        std::vector<float> calc(const std::vector<float> &x);
        void show();
    private:
        std::vector<std::vector<float> > B;
        std::vector<float> C;
};

Gemm::Gemm(const std::string &b_name, const std::string &c_name){
    std::ifstream bf(b_name);
    std::string s;
    while(getline(bf, s)){
        std::stringstream ss(s);
        float d;
        vector<float> vd;
        while(ss>>d){
            vd.push_back(d);
        }
        this->B.push_back(vd);
    }
    ifstream cf(c_name);
    float d;
    vector<float> vd;
    while(cf>>d){
        vd.push_back(d);
    }
    this->C = vd;
}

vector<float> Gemm::calc(const vector<float> &x){
    vector<float> ret = C;
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
        std::vector<float> calc(const std::vector<float> &x);
};

vector<float> Relu::calc(const vector<float> &x){
    vector<float> res = x;
    int n = res.size();
    for(int i=0;i<n;i++){
        res[i] = max(res[i], (float)0.0);
    }
    return res;
}

class MNIST{
    public:
        static const int MNIST_SIZE = 784;
        vector<vector<float>> input;
        vector<int> answer;
        MNIST(const string &f);
        const void show();
        const float accuracy(const vector<int> &x);
};

MNIST::MNIST(const string &f){
    std::ifstream bf(f);
    std::string s;
    while(getline(bf, s)){
        std::stringstream ss(s);
        float d;
        vector<float> vd;
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

const float MNIST::accuracy(const vector<int> &x){
    int n = x.size();
    int c = 0;
    for(int i=0;i<n;i++){
        if(x[i] == this->answer[i]){
            c++;
        }
    }
    return (float)c/(float)n;
}

template<typename T>
const void show_vector(const vector<T> &v){
    for(auto x:v){
        cout<<x<<" ";
    }
    cout<<endl;
}

class Result{
    public:
        Result(double t, double a){
            time = t;
            accuracy = a;
        }
        double time;
        double accuracy;
};

Result original_graph_accuracy(MNIST &mnist, const int ntest){
    Gemm g1("140406444019384_matrix.txt", "140406130172200_matrix.txt");
    Gemm g2("140406443536680_matrix.txt", "140406443536904_matrix.txt");
    Gemm g3("140406443537240_matrix.txt", "140406443537464_matrix.txt");
    Relu r;

    auto start = std::chrono::system_clock::now();
    vector<int> o;
    for(int i=0;i<ntest ;i++){
        vector<float> x = mnist.input[i];
        x = g1.calc(x);
        x = r.calc(x);
        x = g2.calc(x);
        x = r.calc(x);
        x = g3.calc(x);
        int a = distance(x.begin(), max_element(x.begin(), x.end()));
        o.push_back(a);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end-start;
    float ac = mnist.accuracy(o);
    float time = elapsed_seconds.count();
    return Result(time, ac);
}

Result compressed_graph_accuracy(MNIST &mnist, const int ntest, float compress_ratio){
    CompressedGemm g1("140406444019384_matrix.txt", "140406130172200_matrix.txt", compress_ratio);
    CompressedGemm g2("140406443536680_matrix.txt", "140406443536904_matrix.txt", compress_ratio);
    CompressedGemm g3("140406443537240_matrix.txt", "140406443537464_matrix.txt", compress_ratio);
    Relu r;

    auto start = std::chrono::system_clock::now();
    vector<int> o;
    for(int i=0;i<ntest ;i++){
        vector<float> x = mnist.input[i];
        x = g1.calc(x);
        x = r.calc(x);
        x = g2.calc(x);
        x = r.calc(x);
        x = g3.calc(x);
        int a = distance(x.begin(), max_element(x.begin(), x.end()));
        o.push_back(a);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end-start;
    float ac = mnist.accuracy(o);
    float time = elapsed_seconds.count();
    return Result(time, ac);
}

int main(){
    MNIST mnist("mnist_test.txt");

    int n = mnist.answer.size(); 
    cout << "n = " << n << endl;

    {
        cout << "accuracy, time" << endl;
        auto res = original_graph_accuracy(mnist, n);
        cout << setprecision(10) << res.accuracy << ", " << res.time << endl;
    }

    cout << "compress_ratio, accuracy, time" << endl;
    for(int r = 0; r < 80; r+=5){
        auto res = compressed_graph_accuracy(mnist, n, (float)r/100.0);
        cout << setprecision(10) << (float)r/100.0 << ", " << res.accuracy << ", " << res.time << endl;
    }
    for(int r = 80; r <= 100; r+=1){
        auto res = compressed_graph_accuracy(mnist, n, (float)r/100.0);
        cout << setprecision(10) << (float)r/100.0 << ", " << res.accuracy << ", " << res.time << endl;
    }
    return 0;
}
