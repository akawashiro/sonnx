#include <vector>
#include <iomanip>
#include <thread>
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
        void calc_partially(const int index, const vector<float> &x);
        std::vector<float> calc(const std::vector<float> &x);
        void show();
        constexpr static const float DEFAULT_COMPRESS_RATIO = 0.8;
    private:
        int n_thread = 4;
        std::vector<std::vector<float> > B;
        std::vector<float> C;
        std::vector<float> ret;
        vector<float> B_scale;
        vector<int> B_row;
        vector<int> B_column;
        vector<vector<float>> B_scale_threads;
        vector<vector<int>> B_row_threads;
        vector<vector<int>> B_column_threads;
        vector<vector<int>> B_nrows_threads;
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

    // Devide the matrix B to use in threads
    // int row_size = B.size();
    // int column_size = B[0].size();
    int cur=0, cur2=0;
    B_row_threads = vector<vector<int>>(n_thread, vector<int>());
    B_column_threads = vector<vector<int>>(n_thread, vector<int>());
    B_scale_threads = vector<vector<float>>(n_thread, vector<float>());
    B_nrows_threads = vector<vector<int>>(n_thread, vector<int>());
    for(int i=0;i<n_thread;i++){
        while(1){
            if(cur==B_row.size() || (B_row_threads[i].size() > 0 &&
                        B_row_threads[i].back() != B_row[cur])){
                // cout << cur - cur2 << endl;
                B_nrows_threads[i].push_back(cur-cur2);
                cur2 = cur;
            }
            if(cur==B_row.size() || 
                    (B_row_threads[i].size() > B_row.size()/n_thread &&
                     B_row_threads[i].back() != B_row[cur])){
                break;
            }
            B_row_threads[i].push_back(B_row[cur]);
            B_column_threads[i].push_back(B_column[cur]);
            B_scale_threads[i].push_back(B_scale[cur]);
            cur++;
        }
        // cout << B_row_threads[i].size() << endl;
    }
    // cout << "row_size = " << B_row.size() <<endl; 
    // << " column_size = " << column_size << endl;

    ifstream cf(c_name);
    float d;
    vector<float> vd;
    while(cf>>d){
        vd.push_back(d);
    }
    this->C = vd;
}

void CompressedGemm::calc_partially(const int index, const vector<float> &x){
    int m = B_nrows_threads[index].size();
    int cur = 0;
    for(int i=0;i<m;i++){
        int n = B_nrows_threads[index][i];
        float r = 0;
        for(int j=cur;j<cur+n;j++){
            r += B_scale_threads[index][j] * x[B_column_threads[index][j]];
        }
        ret[B_row_threads[index][cur]] += r;
        cur += n;

        // Following code is tyial for vectorization
        // float r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 =0;
        // int j=cur;
        // while(j+7 < cur+n){
        //     r0 += B_scale_threads[index][j] * x[B_column_threads[index][j]];
        //     r1 += B_scale_threads[index][j+1] * x[B_column_threads[index][j+1]];
        //     r2 += B_scale_threads[index][j+2] * x[B_column_threads[index][j+2]];
        //     r3 += B_scale_threads[index][j+3] * x[B_column_threads[index][j+3]];
        //     r4 += B_scale_threads[index][j+4] * x[B_column_threads[index][j+4]];
        //     r5 += B_scale_threads[index][j+5] * x[B_column_threads[index][j+5]];
        //     r6 += B_scale_threads[index][j+6] * x[B_column_threads[index][j+6]];
        //     r7 += B_scale_threads[index][j+7] * x[B_column_threads[index][j+7]];
        //     j+=8;
        // }
        // for(j;j<cur+n;j++){
        //     r1 += B_scale_threads[index][j] * x[B_column_threads[index][j]];
        // }
        // ret[B_row_threads[index][cur]] += (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7);
        // cur += n;
    }

    // int n = B_scale_threads[index].size();
    // float r = 0;
    // for(int i=0;i<n;i++){
    //     r += B_scale_threads[index][i] * x[B_column_threads[index][i]];
    //     if(i<n-1 && B_row_threads[index][i]!=B_row_threads[index][i+1]){
    //         ret[B_row_threads[index][i]] += r;
    //         r = 0;
    //     }
    // }
    // ret[B_row_threads[index][n-1]] += r;
}

vector<float> CompressedGemm::calc(const vector<float> &x){
    ret = C;
    vector<thread> ths;
    
    for(int i=0;i<n_thread;i++){
        ths.push_back(thread(&CompressedGemm::calc_partially, this, i, x));
    }
    for(int i=0;i<n_thread;i++){
        ths[i].join();
    }

    // int n = B_scale.size();
    // for(int i=0;i<n;i++){
    //     ret[B_row[i]] += B_scale[i] * x[B_column[i]];
    // }
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
    // {
    //     cout << "accuracy, time" << endl;
    //     auto res = original_graph_accuracy(mnist, n);
    //     cout << setprecision(10) << res.accuracy << ", " << res.time << endl;
    // }

    cout << "compress_ratio, accuracy, time" << endl;
    // for(int r = 0; r < 80; r+=5){
    //     auto res = compressed_graph_accuracy(mnist, n, (float)r/100.0);
    //     cout << setprecision(10) << (float)r/100.0 << ", " << res.accuracy << ", " << res.time << endl;
    // }
    for(int r = 80; r <= 100; r+=1){
        auto res = compressed_graph_accuracy(mnist, n, (float)r/100.0);
        cout << setprecision(10) << (float)r/100.0 << ", " << res.accuracy << ", " << res.time << endl;
        break;
    }
    return 0;
}
