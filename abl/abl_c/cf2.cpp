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
#include <type_traits> // For std::is_floating_point
#include <cmath> // For std::ceil

using namespace std;

#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) ::std::begin(x), ::std::end(x)
#define ll long long
#define sz(x) (int)(x).size()
#define fo(from_0_to_non_incl_to) for(int i=0;i<from_0_to_non_incl_to;i++)
#define foi(from,non_incl_to) for(int i=from;i<non_incl_to;i++)
#define foj(from,non_incl_to) for(int j=from;j<non_incl_to;j++)
#define fok(from,non_incl_to) for(int k=from;k<non_incl_to;k++)
#define wasd(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define qweasdzxc(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define isvalid(x_plus_i,max_boundary_n,y_plus_j,max_boundary_m) (0<=x_plus_i and x_plus_i<max_boundary_n and 0<=y_plus_j and y_plus_j<max_boundary_m)
//#define gcd __gcd
#define mp make_pair
//Makes % get floor remainder (towards -INF) and make it always positive
#define MOD(x,y) (x%y+y)%y
#define print(p) cout<<p<<endl
#define fi first
#define sec second
#define prmap(m) {for(auto i: m) cout<<(i.fi)<<i.sec<<endl}
#define pra(a) {for(auto i: a) cout<<i<<endl;}
#define prm(a) {for(auto i: a) pra(i) cout<<endl;}
//#define itobin(x) bitset<32> bin(x)
#define itobin(intToConvertTo32BitBinaryNum) std::bitset<32>(intToConvertTo32BitBinaryNum)
#define bintoi(binaryNum32BitToConvertToInt) binaryNum32BitToConvertToInt.to_ulong()
#define binstoi(binaryStringToConvertToInt) stoi(binaryStringToConvertToInt, nullptr, 2)
#define vecsum(vectorName) accumulate((vectorName).begin(), (vectorName).end(), 0)
#define setbits(decimalnumber) __builtin_popcount(decimalnumber)
#define stringSplice(str, i, j) (str).erase(i, j) //j is the length of string to erase starting from index i
#define string_pop_back(str) (str).pop_back()
#define substring(str, i, j) (str).substr(i, j) //j is the length of substring from i

typedef pair<ll, ll> pl;

#define pb push_back

ll INF=1e18;


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

//Graph visualizer:
//https://csacademy.com/app/graph_editor/



int main() {
    std::ios::sync_with_stdio(false);
    setIO("cpp");
    std::cin.tie(nullptr);
    ll t, n;
    cin >> t;
    foi(0,t){
        cin >> n;
        string so,st;
        cin >> so;
        cin >> st;
        vector<vector<ll>> a(2,vector<ll>(n));
        vector<vector<ll>> dp(2,vector<ll>(n,0));
        foj(0,n){
            a[0][j]=so[j]-'0';
            a[1][j]=st[j]-'0';
        }
        //cerr << "og a: " << endl;
        //for(auto x : a) cerr << x << endl;
        dp[0][0]=1;
        if(a[1][0]==0&&a[0][1]==1){
            dp[1][0]=1;
            dp[0][1]=0;
        }
        if(a[1][0]==a[0][1]){
            dp[1][0]=1;
            dp[0][1]=1;
        }
        else{
            dp[0][1]=1;
            dp[1][0]=0;
        }
        ll xt, yt;
        xt=0;
        yt=2;
        ll xb, yb;
        xb=1;
        yb=1;
        //cerr << yt << " " << yb << endl;
        while(yt<n&&yb<n-1){
            //for(auto z : a) cerr << z << endl;
            //cerr << yt << " " << yb << endl;
            if(dp[xt][yt-1]==1){
                if(a[xb][yb]==0&&a[xt][yt]==1){ //i.e. top = 1, bottom = 0 so go bottom
                    //cerr << "reached tt" << endl;
                    //cerr << dp[xb][yb] << endl;
                    dp[xb][yb]=1;
                    dp[xb][yb]+=dp[xb][yb-1];
                    dp[xt][yt]=0;
                }
                else if(a[xb][yb]==a[xt][yt]){ //i.e. top==bottom so go both
                    dp[xb][yb]=1;
                    dp[xb][yb]+=dp[xb][yb-1];
                    dp[xt][yt]=dp[xt][yt-1]; //same as = 1
                }
                else{ //i.e. top = 0, bottom = 1 so go top
                    dp[xt][yt]=1;
                    dp[xb][yb]=0;
                }
            }
            else{
                dp[xt][yt]=0;
                dp[xb][yb]=dp[xb][yb-1];
            }
            yb++;
            yt++;
            //cerr << dp << endl;
        }
        dp[1][n-1]=dp[0][n-1]+dp[1][n-2];
        //cerr << dp << endl;
        //cerr << "reached end" << endl;
        ll x,y;
        y=0;
        x=0;
        string ansss="";
        while(x!=1||y!=n-1){
            //cerr << x << " " << y << endl;
            //for(auto z : a) cerr << z << endl;
            //cerr << endl;
            ansss+=to_string(a[x][y]);
            if(x==0&&y==n-1){
                x=1;
            }
            else{
                if(x==0&&a[1][y]==0&&a[0][y+1]==1){
                    x=1;
                }
                else y++;
            }
            //cerr << x << " " << y << endl;
        }
        ansss+=to_string(a[x][y]);
        cout << ansss << endl;
        cout << dp[1][n-1] << endl;
}
    return 0;
}
