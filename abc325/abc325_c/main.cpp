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

ll mod=1e9+7,INF=1e18;


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

void gen_primes() {
    fill(prime, prime+MAXA, true);
    prime[0] = 0;
    // prime[1] = 1 for our purposes
    for (int i = 4; i < MAXA; i += 2) {
        prime[i] = 0;
    }

    for (int p = 3; p*p < MAXA; p += 2) {
        if (!prime[p]) continue;
        for (int x = p*p; x < MAXA; x += p) {
            prime[x] = 0;
        }
    }
}

//disjoint set union/union find
struct dsu {
  public:
    dsu() : _n(0) {}
    //constructor for dsu. Initialize as "dsu name_of_object(x);"
    explicit dsu(int n) : _n(n), parent_or_size(n, -1) {}

    //returns representative of component if a&b already in component or else joins them into a new component and selects one as representative
    int merge(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        int x = leader(a), y = leader(b);
        if (x == y) return x;
        if (-parent_or_size[x] < -parent_or_size[y]) std::swap(x, y);
        parent_or_size[x] += parent_or_size[y];
        parent_or_size[y] = x;
        return x;
    }

    //returns whether a&b in same component
    bool same(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        return leader(a) == leader(b);
    }

    //returns representative of connected component in which a resides
    int leader(int a) {
        assert(0 <= a && a < _n);
        if (parent_or_size[a] < 0) return a;
        return parent_or_size[a] = leader(parent_or_size[a]);
    }

    //returns size of connected component in which a resides
    int size(int a) {
        assert(0 <= a && a < _n);
        return -parent_or_size[leader(a)];
    }

    //returns a list of the nodes of each connected component
    std::vector<std::vector<int>> groups() {
        std::vector<int> leader_buf(_n), group_size(_n);
        for (int i = 0; i < _n; i++) {
            leader_buf[i] = leader(i);
            group_size[leader_buf[i]]++;
        }
        std::vector<std::vector<int>> result(_n);
        for (int i = 0; i < _n; i++) {
            result[i].reserve(group_size[i]);
        }
        for (int i = 0; i < _n; i++) {
            result[leader_buf[i]].push_back(i);
        }
        result.erase(
            std::remove_if(result.begin(), result.end(),[&](const std::vector<int>& v) { return v.empty(); }),result.end());
        return result;
    }

  private:
    int _n;
    // root node: -1 * component size
    // otherwise: parent
    std::vector<int> parent_or_size;
};


std::vector<std::string> S;
//Graph visualizer:
//https://csacademy.com/app/graph_editor/
vector<vector<bool>> vis;
//dfs traversal from start node, in process keeping track of max depth & keeping track of each node's depth & printing order of traversal
ll maxDepth = 0;
void dfs(ll i, ll j, ll H, ll W){
    vis[i][j] = true;
    S[i][j]='.';
    //cerr << startNode << " ";
    if(isvalid(i+1,H,j,W)){if(S[i+1][j]=='#'&&!vis[i+1][j]) dfs(i+1, j, H, W);}
    if(isvalid(i-1,H,j,W)){ if(S[i-1][j]=='#'&&!vis[i-1][j]) dfs(i-1, j, H, W);}
    if(isvalid(i,H,j+1,W)) {if(S[i][j+1]=='#'&&!vis[i][j+1]) dfs(i, j+1, H, W);}
    if(isvalid(i,H,j-1,W)) {if(S[i][j-1]=='#'&&!vis[i][j-1]) dfs(i, j-1, H, W);}
    if(isvalid(i+1,H,j+1,W)) {if(S[i+1][j+1]=='#'&&!vis[i+1][j+1]) dfs(i+1, j+1, H, W);}
    if(isvalid(i-1,H,j-1,W)) {if(S[i-1][j-1]=='#'&&!vis[i-1][j-1]) dfs(i-1, j-1, H, W);}
    if(isvalid(i+1,H,j-1,W)) {if(S[i+1][j-1]=='#'&&!vis[i+1][j-1]) dfs(i+1, j-1, H, W);}
    if(isvalid(i-1,H,j+1,W)) {if(S[i-1][j+1]=='#'&&!vis[i-1][j+1]) dfs(i-1, j+1, H, W);}
}

long long solve(ll h, long long w, const std::vector<std::string> &S) {
    /* 
    g.assign(n+1, vector<int>());
    wg.assign(n + 1, vector<pair<ll,ll>>());
    parent.assign(n+1, -1); */
    /* ll counter = 0;
    vis.assign(H+1, vector<bool>(W+1, false));
    for(ll i = 0; i<H; i++){
        for(ll j = 0; j<W; j++){
            if(S[i][j]=='#'){
                //cerr << i << " " << j << endl;
                dfs(i,j,H,W);
                //cerr << S << endl;
                counter++;
            }
        }
    }
    return counter; */
    ll n = h*w;
    ll ans = 0;
    dsu uf(n);
    foi(0,h)foj(0,w){
        if(S[i][j] !='#') continue;
        //cerr << "S[i][j]" << i << " " << j << endl;
        //ans++;
        for(ll di = -1; di<=1; di++){
            for(ll dj=-1;dj<=1;dj++){
                ll ni=i+di, nj = j+dj;
                if(ni<0||ni>=h||nj<0||nj>=w) continue;
                if(S[ni][nj]!='#') continue;
                if(i==ni&&j==nj) continue;
                ll v = i*w+j, u=ni*w+nj;
                //cerr << "ni: " << ni << " nj: " << nj << endl;
                //cerr << "v: " << v << " u: " << u << endl;
                if(uf.same(v,u)) continue;
                uf.merge(v,u);
                //ans--;
            }
        }
        //cerr << endl;
    }
    /* for(auto x: uf.groups()){
        cerr << x << endl;
    } */
    set<ll> an;
    foi(0,h)foj(0,w){
        if(S[i][j]=='#'){
            //an.insert(uf.leader(i*w+j));
            if(uf.leader(i*w+j)==i*w+j)ans++;
        }
    }
    //return an.size();
    return ans;
}


int main() {
    std::ios::sync_with_stdio(false);
    setIO("");
    std::cin.tie(nullptr);
    int H;
    long long W;
    std::cin >> H;
    S.resize(H);
    std::cin >> W;
    REP (i, H) {
        std::cin >> S[i];
    }
    auto ans = solve(H, W, S);
    std::cout << ans << '\n';

    /* genprimes(1e5); */

    /* //run the bfs and output order of traversed nodes (for loop is only used for non-connected graphs)
    for (int i = 0; i < n; i++) {
        if (!v[i])
            bfs(i);
    }
    
    //Use for problems where you have to go up,down,left,right. Do x+i & y+j and i&j will test all 4 directions. Do x+i+1 & y+j+1 if 0 indexed
    wasd(
        //cout << "Use this for problems where you have to go up, down, left right" << endl;
    ) */
    return 0;
}
