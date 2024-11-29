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
#include <assert.h>
#include <cassert>
#include <unordered_map>
#include <type_traits> // For std::is_floating_point
#include <cmath> // For std::ceil
#include <iomanip>
#include <unordered_set>
#include <functional>
#include <type_traits>
#include <chrono>
#include <list>
#include <complex>
#include <atcoder/all>

using namespace std;
using namespace atcoder;


/*/---------------------------Looping helpers----------------------/*/
#define rep(i,n) for(int i = 0; i < (n); ++i)
#define repp(i, a, b) for (int i = (a); i < (b); ++i)
#define rrep(i,n) for(int i = 1; i <= (n); ++i)
// #define rep(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define fo(from_0_to_non_incl_to) for(int i=0;i<from_0_to_non_incl_to;i++)
//h CAREFUL if you put an expression as an argument it will give bugs, better assign expression to variable then put that in the foi() as argument
#define foi(from,non_incl_to) for(int i=from;i<(non_incl_to);i++)
#define foii(non_incl_to) for(int i=0;i<(non_incl_to);i++)
#define foj(from,non_incl_to) for(int j=from;j<(non_incl_to);j++)
#define fojj(non_incl_to) for(int j=0;j<(non_incl_to);j++)
#define fok(from,non_incl_to) for(int k=from;k<(non_incl_to);k++)
#define fokk(non_incl_to) for(int k=0;k<(non_incl_to);k++)
#define fol(from,non_incl_to) for(int l=from;l<(non_incl_to);l++)
#define foll(non_incl_to) for(int l=0;l<(non_incl_to);l++)
#define fa(x, dataStructure) for(auto x : dataStructure)
#define fx(dataStructure) for(auto x : dataStructure)

/*/---------------------------Abbreviations----------------------/*/
#define ll long long
#define sz(x) (int)(x).size()
#define fi first
#define sec second
#define se second

#define ALL(x) ::std::begin(x), ::std::end(x)
#define all(a) a.begin(),a.end()
#define rng(a) a.begin(),a.end()
//#define gcd __gcd
#define mp make_pair
#define mt make_tuple
#define pb push_back
ll INF=LLONG_MAX;

/*/---------------------------Data Structures----------------------/*/
typedef pair<ll, ll> pl;
//N.b. next() for next node in list, insert(iterator, value) "inserts" value at position right before iterator
typedef list<ll> dll;
using pll = pair<ll,ll>;
using pi = pair<int,int>;
typedef vector<long long> vll;
typedef std::vector<std::vector<long long>> vvll;

template<typename T> using vc = vector<T>;
template<typename T> using vc = vector<T>;
using vl = vc<ll>;

//max heap priority queue i.e. top() gives largest value
typedef priority_queue<ll> maxpq;
//min heap priority queue i.e. top() gives smallest value
typedef priority_queue<ll, vector<ll>, greater<ll>> minpq;

//multiset provides automatic ordering on insertion but unlike set, keeps duplicate/multiple items of same value
//n.b. set also provides autoamtic ordering on insertion n.b. maps are also sorted automatically on insertion according to key order
//.count(x) O(num_of_x+logN)
//.find(x) O(logN) -> so use find over count if possible
//.insert(x) O(logN) -> inserts s.t. sorted order is maintained
//.erase(x) O(logN)
//begin() O(logN)
typedef multiset<ll> msll;
//doing mymultiset.erase(x) will erase all
#define mserasesingle(mymultiset, x) mymultiset.erase(mymultiset.find(x))
#define mseraseall(mymultiset, x) mymultiset.erase(x)
//find smallest and biggest elements O(1)
#define msmin(mymultiset) *mymultiset.begin()
#define msmax(mymultiset) *mymultiset.rbegin()

/*/---------------------------Misc----------------------/*/
//Makes % get floor remainder (towards -INF) and make it always positive
#define MOD(x,y) (x%y+y)%y
// #define print(p) cout<<p<<endl
#define prmap(m) {for(auto i: m) cout<<(i.fi)<<i.sec<<endl}
#define pra(a) {for(auto i: a) cout<<i<<endl;}
#define prm(a) {for(auto i: a) pra(i) cout<<endl;}
#define vecsum(vectorName) accumulate((vectorName).begin(), (vectorName).end(), 0)
#define isvalid(checking,min_boundary,max_boundary) (0<=checking and checking<max_boundary)

/*/---------------------------Base Conversions----------------------/*/
//#define itobin(x) bitset<32> bin(x)
#define itobin(intToConvertTo32BitBinaryNum) std::bitset<32>(intToConvertTo32BitBinaryNum)
#define bintoi(binaryNum32BitToConvertToInt) binaryNum32BitToConvertToInt.to_ulong()
#define binstoi(binaryStringToConvertToInt) stoi(binaryStringToConvertToInt, nullptr, 2)
#define binstoll(binaryStringToConvertToInt) stoll(binaryStringToConvertToInt, nullptr, 2)

/*/---------------------------Bits----------------------/*/
#define setbits(decimalnumber) __builtin_popcountll(decimalnumber)


/*/---------------------------Strings----------------------/*/
#define stringSplice(str, i, j) (str).erase(i, j) //j is the length of string to erase starting from index i
#define string_pop_back(str) (str).pop_back()
#define substring(str, i, j) (str).substr(i, j) //j is the length of substring from i

/*/---------------------------Custom Hash----------------------/*/
// gp_hash_table<long long, int, custom_hash> safe_hash_table;

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};

/*/---------------------------IO(Debugging)----------------------/*/
template<class T> istream& operator >> (istream &is, vector<T>& V) {
    for(auto &e : V)
        is >> e;
    return is;
}
template <class OStream, class T> OStream &operator<<(OStream &os, const std::vector<T> &vec);
template <class OStream, class T, size_t sz> OStream &operator<<(OStream &os, const std::array<T, sz> &arr);
template <class OStream, class T, class TH> OStream &operator<<(OStream &os, const std::unordered_set<T, TH> &vec);
template <class OStream, class T, class U> OStream &operator<<(OStream &os, const pair<T, U> &pa);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::deque<T> &vec);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::set<T> &vec);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::multiset<T> &vec);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::unordered_multiset<T> &vec);
template <class OStream, class T, class U> OStream &operator<<(OStream &os, const std::pair<T, U> &pa);
template <class OStream, class TK, class TV> OStream &operator<<(OStream &os, const std::map<TK, TV> &mp);
template <class OStream, class TK, class TV, class TH> OStream &operator<<(OStream &os, const std::unordered_map<TK, TV, TH> &mp);
template <class OStream, class... T> OStream &operator<<(OStream &os, const std::tuple<T...> &tpl);

template <class OStream, class T> OStream &operator<<(OStream &os, const std::vector<T> &vec) { os << '['; for (auto v : vec) os << v << ','; os << ']'; return os; }
template <class OStream, class T, size_t sz> OStream &operator<<(OStream &os, const std::array<T, sz> &arr) { os << '['; for (auto v : arr) os << v << ','; os << ']'; return os; }
template <class... T> std::istream &operator>>(std::istream &is, std::tuple<T...> &tpl) { std::apply([&is](auto &&... args) { ((is >> args), ...);}, tpl); return is; }
template <class OStream, class... T> OStream &operator<<(OStream &os, const std::tuple<T...> &tpl) { os << '('; std::apply([&os](auto &&... args) { ((os << args << ','), ...);}, tpl); return os << ')'; }
template <class OStream, class T, class TH> OStream &operator<<(OStream &os, const std::unordered_set<T, TH> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::deque<T> &vec) { os << "deq["; for (auto v : vec) os << v << ','; os << ']'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::set<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::multiset<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::unordered_multiset<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T, class U> OStream &operator<<(OStream &os, const std::pair<T, U> &pa) { return os << '(' << pa.first << ',' << pa.second << ')'; }
template <class OStream, class TK, class TV> OStream &operator<<(OStream &os, const std::map<TK, TV> &mp) { os << '{'; for (auto v : mp) os << v.first << "=>" << v.second << ','; os << '}'; return os; }
template <class OStream, class TK, class TV, class TH> OStream &operator<<(OStream &os, const std::unordered_map<TK, TV, TH> &mp) { os << '{'; for (auto v : mp) os << v.first << "=>" << v.second << ','; os << '}'; return os; }


void setIO(string name = "")
{ // name is nonempty for USACO file I/O
    ios_base::sync_with_stdio(0);
    cin.tie(0); // see Fast Input & Output
    // alternatively, cin.tie(0)->sync_with_stdio(0);
    if (sz(name))
    {
    freopen((name + ".in").c_str(), "r", stdin); // see Input & Output
    freopen((name + ".out").c_str(), "w", stdout);
    }
}

/*/---------------------------Custom library - most used only----------------------/*/


/*/---------------------------Syntax hints for mint once import mint.cpp----------------------/*/
//n.b. it is a data type so declare variablesas: mint x;
// to convert any other data type such as int or ll to mint, do: mint(x);
// when you want to access the value of a mint, use x.val()
// e.g. modint998244353 a = modint998244353(x); // `a` now represents `x` modulo 998244353
// using mint = modint998244353;
// Custom operator<< for modint998244353
// How to use the ACL modular exponentiation function?
// e.g. to do pow(10,6)
// mint(10).pow(6)

// //uncomment this code to allow dbg / ostream to handle mint
// std::ostream& operator<<(std::ostream& os, const mint& m) {
//     return os << m.val();
// }

#ifdef isym444_LOCAL
const string COLOR_RESET = "\033[0m", BRIGHT_GREEN = "\033[1;32m", BRIGHT_RED = "\033[1;31m", BRIGHT_CYAN = "\033[1;36m", NORMAL_CROSSED = "\033[0;9;37m", RED_BACKGROUND = "\033[1;41m", NORMAL_FAINT = "\033[0;2m";
#define dbg(x) std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << COLOR_RESET << std::endl
#define dbgif(cond, x) ((cond) ? std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << __FILE__ << COLOR_RESET << std::endl : std::cerr)
#else
#define dbg(x) ((void)0)
#define dbgif(cond, x) ((void)0)
#endif

ll midpoint(ll L, ll R){
    return (L+(R-L)/2);
}

template<typename T, typename U>
auto ceildiv(T n, U d) -> decltype(n / d + 0) {
    static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, "ceildiv requires arithmetic types");

    if constexpr (std::is_floating_point<T>::value || std::is_floating_point<U>::value) {
        // Handle case where either n or d is a floating-point number
        return static_cast<decltype(n / d + 0)>(std::ceil(n / static_cast<double>(d)));
    } else {
        // Handle case where both n and d are integers
        return (n + d - 1) / d;
    }
}

/* ll ceildiv(ll n, ll d){
return((n+d-1)/d);
}
*/
/* ll floordiv(ll n, ll d){
ll x = (n%d+d)%d;
return ((n-x)/d);
}
 */
template<typename T, typename U>
auto floordiv(T n, U d) -> decltype(n / d + 0) {
    static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, "floordiv requires arithmetic types");

    if constexpr (std::is_floating_point<T>::value || std::is_floating_point<U>::value) {
        // Handle case where either n or d is a floating-point number
        // Perform the division as floating-point operation and use std::floor to round down
        return static_cast<decltype(n / d + 0)>(std::floor(n / static_cast<double>(d)));
    } else {
        // Handle case where both n and d are integers
        // Original logic for floor division with integers
        T x = (n % d + d) % d;
        return (n - x) / d;
    }
}

template <class T> std::vector<T> sort_unique(std::vector<T> vec) { sort(vec.begin(), vec.end()), vec.erase(unique(vec.begin(), vec.end()), vec.end()); return vec; }
//index of the first occurrence of x. If x is not present in the vector, it returns the index where x can be inserted while keeping the vector sorted
template <class T> int indlb(const std::vector<T> &v, const T &x) { return std::distance(v.begin(), std::lower_bound(v.begin(), v.end(), x)); }
//index immediately after the last occurrence of x. If x is not present, like the lower bound, it returns the index where x can be inserted to maintain order
template <class T> int indub(const std::vector<T> &v, const T &x) { return std::distance(v.begin(), std::upper_bound(v.begin(), v.end(), x)); }

/*/---------------------------Useful Graph Visualizer----------------------/*/
//https://csacademy.com/app/graph_editor/

//h INSERT CODE SNIPPETS HERE

long long ipow(long long base, int exp) {
    long long result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}



/*/---------------------------INSERT CODE SNIPPETS HERE----------------------/*/

#define nyan ios::sync_with_stdio(false);cin.tie(nullptr);cout<<fixed<<setprecision(15);
struct custom{
    double simpleSum;
    double numeratorOnly;
    double R;
};

int main(){
    nyan;
    ll N;
    cin >> N;
    vll P(N);
    cin >> P;
    // dbg(P);
    // double pow09 = 1.0L;

    vector<double> denom(N+2);
    vector<double> RHS(N+2);
    denom[1]=(double)1.0;
    foi(2,N+1){
        denom[i]=denom[i-1]+pow(0.9L,i-1);
    }

    // vector<double> denom(N + 2, 1.0L); // denom[1] = 1.0
    // for (ll i = 2; i <= N + 1; ++i) {
    //     denom[i] = denom[i - 1] * 0.9L + 1.0L;
    // }
    // for (int i = 1; i <= N+1; ++i) {
    //     denom[i] = denom[i-1] + pow09;
    //     pow09 *= 0.9L;
    // }
    foi(1,N+1){
        RHS[i]=(double)1200.0L/sqrtl(i);
    }
    // dbg(denom);
    // dbg(RHS);
    vector<vector<custom>> dp(N+1,vector<custom>(N+1,{0,0,(double)-INF}));
    // vector<vector<custom>> dp(N+1,vector<custom>(N+1,{0,0,(double)-1e18}));
    // dp[0][0]={0.0L,0.0L,0.0L};
    // custom t1 = {2,3,4};
    // dbg(t1.numeratorOnly);

    foj(0,N){
        foi(0,j+1){
            // don't select
            // dp[i][j+1].R=max(dp[i][j].R,dp[i][j+1].R);
            if(dp[i][j+1].R<dp[i][j].R){
                dp[i][j+1]=dp[i][j];
            }
            
            // do select
            double oldR = dp[i+1][j+1].R;
            double oldNumerator = dp[i][j].numeratorOnly;
            double oldSimpleSum = dp[i][j].simpleSum;
            double newNumerator = (double)0.9L*oldNumerator+P[j];
            double newSimpleSum = oldSimpleSum+P[j];
            double newR = newNumerator/denom[i+1]-RHS[i+1];
            // dbg(mt(newNumerator, newSimpleSum, denom[i+1], RHS[i+1], newR));
            if(newR>oldR){
                dp[i+1][j+1]={newSimpleSum, newNumerator, newR};
            }
            // dp[i+1][j+1].R=max(dp[i][j].R+xxxxx,dp[i+1][j+1].R);
            
        }
    }
    double ans=-INF;
    // double ans=(double)-1e18;
    foi(0,N+1){
        foj(0,N+1){
            double cur = dp[i][j].R;
            // dbg(cur);
            ans=max(ans,cur);
        }
    }
    cout << ans << endl;
    return 0;
}
