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


vector<ll> parent; // To store parent information
//visited nodes
vector<bool> vis;
//bool vis[61][61][61][61]={0};
map<ll,ll> depth;

//initialize graph as adjacency list
vector<vector<ll> > g;
//initialize weighted graph as adjacency list
vector<vector<pair<ll,ll>>> dg;
//for building the adjacency list by adding edges info
void edge(ll originNode, ll destNode)
{
    g[originNode].pb(destNode);
 
    // for undirected graph e.g. tree, add this line:
    // g[destNode].pb(originNode);
}

void edge(ll originNode, ll destNode, ll weight){
    dg[originNode].emplace_back(destNode, weight);
    // For an undirected graph e.g., tree, add this line:
    // g[destNode].emplace_back(originNode, weight);
}

//returns vector where each index is the shortest distance between the start node and node i
vector<ll> dijkstra(int start) {
    vector<ll> dist(dg.size(), INF);  // Distance from start to each node
    //arguments: 1) type of elements pq will store 2) underlying container to be used by pq 
    //3) comparison function to specify order of elements in pq (default is less with largest element at top i.e. max-heap vs min-heap below)
    priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, greater<pair<ll, ll>>> pq;
    dist[start] = 0;
    pq.push({0, start});  // {distance, node}

    while (!pq.empty()) {
        //cerr << "pq" << pq << endl;
        ll currentDist = pq.top().first;
        ll currentNode = pq.top().second;
        pq.pop();

        // If the distance in priority queue is larger, we have already found a better path
        if (currentDist > dist[currentNode]) {
            continue;
        }

        for (auto &neighbor : dg[currentNode]) {
            ll nextNode = neighbor.first;
            ll weight = neighbor.second;
            ll newDist = currentDist + weight;

            if (newDist < dist[nextNode]) {
                dist[nextNode] = newDist;
                pq.push({newDist, nextNode});
            }
        }
    }

    return dist;
}

//Graph visualizer:
//https://csacademy.com/app/graph_editor/



long long solve(int N, const std::vector<long long> &A, const std::vector<long long> &B, const std::vector<long long> &X) {
    vis.assign(N+1, false);
    dg.assign(N + 1, vector<pair<ll,ll>>());
    parent.assign(N+1, -1);
    for(ll i = 0; i<A.size(); i++){
        edge(i+1, i+2, A[i]);
        edge(i+1,X[i],B[i]);
    }
    for(auto DG : dg){
        cerr << DG << endl;
    }
    vector<ll> potans;
    potans = dijkstra(1);
    //cerr << "potans: " << potans << endl;
    return potans[N];
}

int main() {
    std::ios::sync_with_stdio(false);
    setIO("");
    std::cin.tie(nullptr);
    int N;
    std::cin >> N;
    
    std::vector<long long> A(N - 1), B(N - 1), X(N - 1);
    REP (i, N - 1) {
        std::cin >> A[i] >> B[i] >> X[i];
    }
    auto ans = solve(N, A, B, X);
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
