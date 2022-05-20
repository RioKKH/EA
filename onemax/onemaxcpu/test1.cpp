#include <iostream>
#include <vector>
#include <string>
#include <numeric>

int main()
{
    const std::vector<int> v = {1, 2, 3, 4, 5};
    const std::vector<std::string> v2 = {"aaa", "bbb", "ccc"};

    // (1) : 合計値を求める
    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "sum : " << sum << std::endl;

    // (1) : 合計値をlong long型として求める
    // accumulateの第3引数がlong long型のゼロを表す0LLになっていることに注意
    // accumulateの戻り値型は、第3引数の型となるため、変数sum_llの型はlong long
    auto sum_ll = std::accumulate(v.begin(), v.end(), 0LL);
    std::cout << "sum_ll : " << sum_ll << std::endl;

    // (1) : 文字列のリストを連結する
    std::string concatenate = std::accumulate(v2.begin(), v2.end(), std::string());
    std::cout << "concat : " << concatenate << std::endl;

    // (2) : 任意の二項演算を行う
    // ここでは、初期値を1として、すべての要素を掛け合わせている
    int product = std::accumulate(v.begin(), v.end(), 1, [](int acc, int i) {
            return acc * i;
    });

    std::cout << "product : " << product << std::endl;
}
