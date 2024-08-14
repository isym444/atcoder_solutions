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

//graph as adjacency list
vector<vector<ll> > g;
void edge(ll originNode, ll destNode)
{
    g[originNode].pb(destNode);
 
    // for undirected graph e.g. tree, add this line
    // g[destNode].pb(originNode);
}

//traverse a graph using bfs from the specified start node to all other nodes, in the process printing the order the nodes are visited in
void bfs(ll start)
{
    queue<ll> q;
 
    q.push(start);
    vis[start] = true;
    depth[start] = 1; // Depth of starting node is 1
    //If want first time something happens/node reached then break when that happens
    while (!q.empty()) {
        ll f = q.front();
        q.pop();
 
        cerr << f << " ";
        
        // Enqueue all adjacent of f and mark them visited 
        ll counter = 0;
        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!vis[*i]) {
                counter++;
                q.push(*i);
                vis[*i] = true;
                depth[*i] = depth[f] + 1; // Set the depth of the neighboring node
            }
        }
        /* if(counter==0){
            cerr << "depths to leafs: " << depth[f] << endl;
        } */
    }
}

//return a vector containing bfs path from start to end nodes specified
vector<ll> bfs(ll start, ll end) {
    queue<ll> q;
    q.push(start);
    vis[start] = true;
    parent[start] = -1; // Start node has no parent

    while (!q.empty()) {
        ll f = q.front();
        q.pop();

        if (f == end) break; // Stop if we reach the end node

        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!vis[*i]) {
                q.push(*i);
                vis[*i] = true;
                parent[*i] = f; // Set parent
            }
        }
    }

    vector<ll> path;
    for (ll i = end; i != -1; i = parent[i]) {
        path.push_back(i);
    }
    reverse(path.begin(), path.end()); // Reverse to get the correct order
    return path;
}

//dfs traversal from start node, in process keeping track of max depth & keeping track of each node's depth & printing order of traversal
ll maxDepth = 0;
void dfs(ll startNode, ll startDepth){
    vis[startNode] = true;
    depth[startNode]=startDepth;
    maxDepth=max(maxDepth, startDepth);
    cerr << startNode << " ";
    for(auto adjNode : g[startNode]){
        if(!vis[adjNode]) dfs(adjNode, startDepth+1);
    }
}

map<ll,ll>subtreeSizes; //Map to store subtree sizes for each child of the start node
ll dfsSubtreesHelper(ll startNode){
    vis[startNode] = true;
    ll subtreeSize = 1;
    //cerr << startNode << " ";
    for(auto adjNode : g[startNode]){
        if(!vis[adjNode]){
            subtreeSize+=dfsSubtreesHelper(adjNode);
        }
    }
    return subtreeSize;
}
//main function to call to populate subtreeSizes
ll minSubtreeSize = 3*pow(10,5)+1; //Adjust this to the max given boundary of the problem
void dfsSubtrees(ll startNode){
    vis[startNode] = true;
    for(auto adjNode : g[startNode]){
        subtreeSizes[adjNode]=dfsSubtreesHelper(adjNode); //+1 if want to include startNode in size of subtrees
        minSubtreeSize=min(minSubtreeSize,subtreeSizes[adjNode]);
    }
}

string itobins(int n) {
    if (n == 0) return "0";

    string binary = "";
    while (n > 0) {
        binary += (n % 2) ? '1' : '0';
        n /= 2;
    }

    reverse(binary.begin(), binary.end()); // Reverse to get the correct order
    return binary;
}

string dtobx(int decimalNumber, int base) {
    if (base < 2 || base > 36) {
        return "Invalid base";
    }

    string result = "";
    while (decimalNumber > 0) {
        int remainder = decimalNumber % base;

        // Convert remainder to corresponding character
        if (remainder >= 10) {
            result += 'A' + (remainder - 10);
        } else {
            result += '0' + remainder;
        }

        decimalNumber /= base;
    }

    // Reverse the string as the result is calculated in reverse order
    reverse(result.begin(), result.end());

    return result.empty() ? "0" : result;
}

ll ceildiv(ll n, ll d){
    return((n+d-1)/d);
}

ll floordiv(ll n, ll d){
    ll x = (n%d+d)%d;
    return ((n-x)/d);
}

ll midpoint(ll L, ll R){
    return (L+(R-L)/2);
}

ll lcm(ll a, ll b) {
    return std::abs(a * b) / std::gcd(a, b);
}


int stringToBinary(const std::string& s, char charAsOne) {
    int x = 0;
    for (int j = 0; j < s.length(); j++) {
        x = 2 * x + (s[j] == charAsOne);
    }
    return x;
}

//returns index of first element greater than or equal to target
ll findGreaterEqual(vector<ll> sortedVector, ll target){
    auto it = lower_bound(sortedVector.begin(), sortedVector.end(), target);
    return it-sortedVector.begin();
}

//returns index of first element less than or equal to target
//if all elements are greater than target returns -1
//if all elements are smaller than target, returns last element
ll findLessEqual(vector<ll> sortedVector, ll target){
    auto it = upper_bound(sortedVector.begin(), sortedVector.end(), target);
    if(it != sortedVector.begin()){
        --it;
        if(*it<=target){
            return it-sortedVector.begin()+1;
        }
    }
    else{
        return -1;
    }
}

struct loc
{
    inline static ll x=0;
    inline static ll y=0;
    inline static char dir='r';
    //loc::x to access or modify x
};

//Graph visualizer:
//https://csacademy.com/app/graph_editor/

#include<cassert>
#include<atcoder/lazysegtree>

long op(long f,long x){return f+x;}       // Defines the operation for the segment tree (sum operation).
long e(){return 0L;}                      // Defines the identity element for the segment tree (0 for sum).

int N,M;

int main()
{
    ios::sync_with_stdio(false);          // Speeds up I/O operations.
    cin.tie(nullptr);                     // Unlinks the input and output streams.

    cin>>N>>M;                            // Read the number of boxes (N) and number of operations (M).
    vector<long>A(N);                     // Create a vector to hold the initial number of balls in each box.
    cin  >> A;                            // Read the initial number of balls in each box.

    atcoder::lazy_segtree<long,op,e,long,op,op,e>seg(A);
                                          // Initialize a lazy segment tree with the array A.

    for(int i=0;i<M;i++)                  // Process each operation.
    {
        int b; cin>>b;                    // Read the index `b` (the box to pick up balls from).
        long c = seg.get(b);              // Get the current number of balls in box `b`.
        seg.set(b, 0L);                   // Set the number of balls in box `b` to 0.

        seg.apply(0, N, c/N);             // Distribute `c/N` balls evenly across all boxes.

        c %= N;                           // Calculate the remainder `c` after even distribution.
        if(c == 0) continue;              // If no remainder, skip to the next operation.

        if(b + c < N)
            seg.apply(b + 1, b + c + 1, 1L);
                                          // Distribute the remaining `c` balls in a straightforward range.
        else
        {
            seg.apply(b + 1, N, 1L);      // If the range crosses the end, handle the overflow separately.
            seg.apply(0, b + c - N + 1, 1L);
        }
    }

    for(int i = 0; i < N; i++)            // Output the final state of all boxes.
        cout << seg.get(i) << (i + 1 == N ? "\n" : " ");
}