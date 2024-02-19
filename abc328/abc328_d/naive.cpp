//#define _GLIBCXX_DEBUG

//#pragma GCC target("avx2")
//#pragma GCC optimize("O3")
//#pragma GCC optimize("unroll-loops")

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <queue>
#include <array>
#include <climits>
#include <cmath>
#include <set>
#include <map>
#include <bitset>
#include <deque>
#include <numeric>
#include <stack>
#ifdef NACHIA
// #define _GLIBCXX_DEBUG
#else
#define NDEBUG
#endif
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>
#include <array>
#include <cmath>
using namespace std;
using i64 = long long;
using u64 = unsigned long long;
#define rep(i,n) for(i64 i=0; i<(i64)(n); i++)
#define repr(i,n) for(i64 i=(i64)(n)-1; i>=0; i--)
const i64 INF = 1001001001001001001;
const char* yn(bool x){ return x ? "Yes" : "No"; }
template<typename A> void chmin(A& l, const A& r){ if(r < l) l = r; }
template<typename A> void chmax(A& l, const A& r){ if(l < r) l = r; }
template<typename A> using nega_queue = priority_queue<A,vector<A>,greater<A>>;

#include <iterator>
#include <functional>

template<class Elem> struct vec;

template<class Iter>
struct seq_view{
    using Ref = typename std::iterator_traits<Iter>::reference;
    using Elem = typename std::iterator_traits<Iter>::value_type;
    Iter a, b;
    Iter begin() const { return a; }
    Iter end() const { return b; }
    int size() const { return (int)(b-a); }
    seq_view(Iter first, Iter last) : a(first), b(last) {}
    seq_view sort() const { std::sort(a, b); return *this; }
    Ref& operator[](int x){ return *(a+x); }
    template<class F = std::less<Elem>, class ret = vec<int>> ret sorti(F f = F()) const {
        ret x(size()); for(int i=0; i<size(); i++) x[i] = i;
        x().sort([&](int l, int r){ return f(a[l],a[r]); });
        return x;
    }
    template<class ret = vec<Elem>> ret col() const { return ret(begin(), end()); }
    template<class F = std::equal_to<Elem>, class ret = vec<std::pair<Elem, int>>>
    ret rle(F eq = F()) const {
        auto x = ret();
        for(auto& a : (*this)){
            if(x.size() == 0 || !eq(x[x.size()-1].first, a)) x.emp(a, 1); else x[x.size()-1].second++;
        } return x;
    }
    template<class F> seq_view sort(F f) const { std::sort(a, b, f); return *this; }
    Iter uni() const { return std::unique(a, b); }
    Iter lb(const Elem& x) const { return std::lower_bound(a, b, x); }
    Iter ub(const Elem& x) const { return std::upper_bound(a, b, x); }
    int lbi(const Elem& x) const { return lb(x) - a; }
    int ubi(const Elem& x) const { return ub(x) - a; }
    seq_view bound(const Elem& l, const Elem& r) const { return { lb(l), lb(r) }; }
    template<class F> Iter lb(const Elem& x, F f) const { return std::lower_bound(a, b, x, f); }
    template<class F> Iter ub(const Elem& x, F f) const { return std::upper_bound(a, b, x, f); }
    template<class F> Iter when_true_to_false(F f) const {
        if(a == b) return a;
        return std::lower_bound(a, b, *a,
            [&](const Elem& x, const Elem&){ return f(x); });
    }
    seq_view same(Elem x) const { return { lb(x), ub(x) }; }
    template<class F> auto map(F f) const {
        vec<typename Iter::value_type> r;
        for(auto& x : *this) r.emp(f(x));
        return r;
    }
    Iter max() const { return std::max_element(a, b); }
    Iter min() const { return std::min_element(a, b); }
    template<class F = std::less<Elem>>
    Iter min(F f) const { return std::min_element(a, b, f); }
    seq_view rev() const { std::reverse(a, b); return *this; }
};

template<class Elem>
struct vec {
    using Base = typename std::vector<Elem>;
    using Iter = typename Base::iterator;
    using CIter = typename Base::const_iterator;
    using View = seq_view<Iter>;
    using CView = seq_view<CIter>;

    vec(){}
    explicit vec(int n, const Elem& value = Elem()) : a(0<n?n:0, value) {}
    template <class I2> vec(I2 first, I2 last) : a(first, last) {}
    vec(std::initializer_list<Elem> il) : a(std::move(il)) {}
    vec(Base b) : a(std::move(b)) {}
    operator Base() const { return a; }

    Iter begin(){ return a.begin(); }
    CIter begin() const { return a.begin(); }
    Iter end(){ return a.end(); }
    CIter end() const { return a.end(); }
    int size() const { return a.size(); }
    bool empty() const { return a.empty(); }
    Elem& back(){ return a.back(); }
    const Elem& back() const { return a.back(); }
    vec sortunied(){ vec x = *this; x().sort(); x.a.erase(x().uni(), x.end()); return x; }
    Iter operator()(int x){ return a.begin() + x; }
    CIter operator()(int x) const { return a.begin() + x; }
    View operator()(int l, int r){ return { (*this)(l), (*this)(r) }; }
    CView operator()(int l, int r) const { return { (*this)(l), (*this)(r) }; }
    View operator()(){ return (*this)(0,size()); }
    CView operator()() const { return (*this)(0,size()); }
    Elem& operator[](int x){ return *((*this)(x)); }
    const Elem& operator[](int x) const { return *((*this)(x)); }
    Base& operator*(){ return a; }
    const Base& operator*() const { return a; }
    vec& push(Elem args){
        a.push_back(std::move(args));
        return *this;
    }
    template<class... Args>
    vec& emp(Args &&... args){
        a.emplace_back(std::forward<Args>(args) ...);
        return *this;
    }
    template<class Range>
    vec& app(Range& x){ for(auto& v : a) emp(v); }
    Elem pop(){
        Elem x = std::move(a.back());
        a.pop_back(); return x;
    }
    bool operator==(const vec& r) const { return a == r.a; }
    bool operator!=(const vec& r) const { return a != r.a; }
    bool operator<(const vec& r) const { return a < r.a; }
    bool operator<=(const vec& r) const { return a <= r.a; }
    bool operator>(const vec& r) const { return a > r.a; }
    bool operator>=(const vec& r) const { return a >= r.a; }
    vec<vec<Elem>> pile(int n) const { return vec<vec<Elem>>(n, *this); }
private: Base a;
};

template<class IStr, class U, class T>
IStr& operator>>(IStr& is, vec<std::pair<U,T>>& v){ for(auto& x:v){ is >> x.first >> x.second; } return is; }
template<class IStr, class T>
IStr& operator>>(IStr& is, vec<T>& v){ for(auto& x:v){ is >> x; } return is; }
template<class OStr, class T>
OStr& operator<<(OStr& os, const vec<T>& v){
    for(int i=0; i<v.size(); i++){
        if(i){ os << ' '; } os << v[i];
    } return os;
}

void testcase(){
    string S; cin >> S;
    string buf;
    for(char c : S){
        buf.push_back(c);
        if(buf.size() >= (size_t)3 && buf.substr(buf.size()-3) == "ABC"){
            rep(t,3) buf.pop_back();
        }
    }
    cout << buf << endl;
}

int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    #ifdef NACHIA
    int T; cin >> T; for(int t=0; t<T; T!=++t?(cout<<'\n'),0:0)
    #endif
    testcase();
    return 0;
}
