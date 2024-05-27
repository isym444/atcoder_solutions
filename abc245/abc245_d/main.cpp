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
#include <unordered_map>

using namespace std;

#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) ::std::begin(x), ::std::end(x)
#define ll long long
#define sz(x) (int)(x).size()
#define fo(from_0_to_non_incl_to) for(int i=0;i<from_0_to_non_incl_to;i++)
//h CAREFUL if you put an expression as an argument it will give bugs, better assign expression to variable then put that in the foi() as argument
#define foi(from,non_incl_to) for(int i=from;i<(non_incl_to);i++)
#define foii(non_incl_to) for(int i=0;i<(non_incl_to);i++)
#define foj(from,non_incl_to) for(int j=from;j<(non_incl_to);j++)
#define fojj(non_incl_to) for(int j=0;j<(non_incl_to);j++)
#define fok(from,non_incl_to) for(int k=from;k<(non_incl_to);k++)
#define fokk(non_incl_to) for(int k=0;k<(non_incl_to);k++)
#define fa(x, dataStructure) for(auto x : dataStructure)
#define fx(dataStructure) for(auto x : dataStructure)
#define wasd(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define qweasdzxc(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
// #define isvalid(x_plus_i,max_boundary_n,y_plus_j,max_boundary_m) (0<=x_plus_i and x_plus_i<max_boundary_n and 0<=y_plus_j and y_plus_j<max_boundary_m)
#define isvalid(x_value,y_value,min_valid_x,max_valid_x,min_valid_y,max_valid_y) (min_valid_x<=x_value and x_value<=max_valid_x and min_valid_y<=y_value and y_value<=max_valid_y)
//#define gcd __gcd
#define mp make_pair
#define mt make_tuple
//Makes % get floor remainder (towards -INF) and make it always positive
#define MOD(x,y) (x%y+y)%y
// #define print(p) cout<<p<<endl
#define fi first
#define sec second
#define prmap(m) {for(auto i: m) cout<<(i.fi)<<i.sec<<endl}
#define pra(a) {for(auto i: a) cout<<i<<endl;}
#define prm(a) {for(auto i: a) pra(i) cout<<endl;}
//#define itobin(x) bitset<32> bin(x)
#define itobin(intToConvertTo32BitBinaryNum) std::bitset<32>(intToConvertTo32BitBinaryNum)
#define bintoi(binaryNum32BitToConvertToInt) binaryNum32BitToConvertToInt.to_ulong()
#define binstoi(binaryStringToConvertToInt) stoi(binaryStringToConvertToInt, nullptr, 2)
#define binstoll(binaryStringToConvertToInt) stoll(binaryStringToConvertToInt, nullptr, 2)
#define vecsum(vectorName) accumulate((vectorName).begin(), (vectorName).end(), 0)
#define setbits(decimalnumber) __builtin_popcount(decimalnumber)
#define stringSplice(str, i, j) (str).erase(i, j) //j is the length of string to erase starting from index i
#define string_pop_back(str) (str).pop_back()
#define substring(str, i, j) (str).substr(i, j) //j is the length of substring from i
#define rng(a) a.begin(),a.end()
#define all(a) a.begin(),a.end()

typedef pair<ll, ll> pl;
typedef vector<long long> vll;
typedef std::vector<std::vector<long long>> vvll;

#define pb push_back

//max heap priority queue i.e. top() gives largest value
typedef priority_queue<ll> maxpq;
//min heap priority queue i.e. top() gives smallest value
typedef priority_queue<ll, vector<ll>, greater<ll>> minpq;

//multiset provides automatic ordering on insertion but unlike set, keeps duplicate/multiple items of same value
//n.b. set also provides autoamtic ordering on insertion
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

int digit_to_int(char c) { return c - '0'; }
int lowercase_to_int(char c) { return c - 'a'; }
int uppercase_to_int(char c) { return c - 'A'; }
ll INF=LLONG_MAX;


// gp_hash_table<long long, int, custom_hash> safe_hash_table;


/*/---------------------------IO(Debugging)----------------------/*/
template<class T> istream& operator >> (istream &is, vector<T>& V) {
    for(auto &e : V)
        is >> e;
    return is;
}

template<typename CharT, typename Traits, typename T>
ostream& _containerprint(std::basic_ostream<CharT, Traits> &out, T const &val) {
    return (out << val << " ");
}
template<typename CharT, typename Traits, typename T1, typename T2>
ostream& _containerprint(std::basic_ostream<CharT, Traits> &out, pair<T1, T2> const &val) {
    return (out << "(" << val.first << "," << val.second << ") ");
}
template<typename CharT, typename Traits, template<typename, typename...> class TT, typename... Args>
ostream& operator << (std::basic_ostream<CharT, Traits> &out, TT<Args...> const &cont) {
    out << "[ ";
    for(auto&& elem : cont) _containerprint(out, elem);
    return (out << "]");
}
template<class L, class R> ostream& operator << (ostream& out, pair<L, R> const &val){
    return (out << "(" << val.first << "," << val.second << ") ");
}
template<class P, class Q = vector<P>, class R = less<P> > ostream& operator << (ostream& out, priority_queue<P, Q, R> const& M){
    static priority_queue<P, Q, R> U;
    U = M;
    out << "{ ";
    while(!U.empty())
        out << U.top() << " ", U.pop();
    return (out << "}");
}
template<class P> ostream& operator << (ostream& out, queue<P> const& M){
    static queue<P> U;
    U = M;
    out << "{ ";
    while(!U.empty())
        out << U.front() << " ", U.pop();
    return (out << "}");
}
template<typename CharT, typename Traits>
ostream& operator << (std::basic_ostream<CharT, Traits> &out, vector<vector<ll>> const &matrix) {
    for (auto &row : matrix) {
        out << row << "\n";
    }
    return out;
}

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


const int MAXA = 5000006;
bool prime[MAXA];


#ifdef isym444_LOCAL
const string COLOR_RESET = "\033[0m", BRIGHT_GREEN = "\033[1;32m", BRIGHT_RED = "\033[1;31m", BRIGHT_CYAN = "\033[1;36m", NORMAL_CROSSED = "\033[0;9;37m", RED_BACKGROUND = "\033[1;41m", NORMAL_FAINT = "\033[0;2m";
#define dbg(x) std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << COLOR_RESET << std::endl
#define dbgif(cond, x) ((cond) ? std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << __FILE__ << COLOR_RESET << std::endl : std::cerr)
#else
#define dbg(x) ((void)0)
#define dbgif(cond, x) ((void)0)
#endif


//Graph visualizer:
//https://csacademy.com/app/graph_editor/

//n.b. it is a data type so declare variablesas: mint x;
// to convert any other data type such as int or ll to mint, do: mint(x);
// when you want to access the value of a mint, use x.val()
// e.g. modint998244353 a = modint998244353(x); // `a` now represents `x` modulo 998244353

//C / A = B where each is a vector representing coefficients of polynomial including the x^0 coefficient i.e. the c in mx+c
vector<long long> polynomialDivision(const vector<long long>& c, const vector<long long>& a) {
    int n = a.size() - 1;  // Degree of A(x)
    int m = c.size() - n - 1;  // Degree of B(x)
    vector<long long> b(m + 1);
    vector<long long> temp_c = c;  // Copy of c to modify during the process

    for (int i = m; i >= 0; --i) {
        b[i] = temp_c[i + n] / a[n];
        for (int j = 0; j <= n; ++j) {
            temp_c[i + j] -= b[i] * a[j];
        }
    }

    return b;
}

int main(){
    ll N, M;
    cin >> N >> M;
    vll A(N+1);
    cin >> A;
    vll C(N+M+1);
    cin >> C;
    dbg(A);
    dbg(C);
    vll B(M+1);
    B = polynomialDivision(C,A);
    fx(B){
        cout << x << " ";
    }
    return 0;
}