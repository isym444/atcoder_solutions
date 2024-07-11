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
#include <cassert>
#include <functional>

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
#define gcd __gcd
#define mp make_pair
#define MOD(x,y) (x%y+y)%y
#define print(p) cout<<p<<endl;
#define fi first
#define sec second
#define prmap(m) {for(auto i: m) cout<<(i.fi)<<i.sec<<endl}
#define pra(a) {for(auto i: a) cout<<i<<endl;}
#define prm(a) {for(auto i: a) pra(i) cout<<endl;}
//#define itobin(x) bitset<32> bin(x)
#define itobin(variable, x) std::bitset<32> variable(x)
#define bintoi(x) x.to_ulong()
#define binstoi(x) stoi(x, nullptr, 2)
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

//visited nodes
vector<bool> v;
//bool vis[61][61][61][61]={0};
//graph as adjacency list
vector<vector<int> > g;

void edge(int a, int b)
{
    g[a].pb(b);
 
    // for undirected graph add this line
    // g[b].pb(a);
}

void bfs(int u)
{
    queue<int> q;
 
    q.push(u);
    v[u] = true;
 
    //If want first time something happens/node reached then break when that happens
    while (!q.empty()) {
 
        int f = q.front();
        q.pop();
 
        cout << f << " ";
 
        // Enqueue all adjacent of f and mark them visited 
        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!v[*i]) {
                q.push(*i);
                v[*i] = true;
            }
        }
    }
}

vector<int> parent; // To store parent information

vector<int> bfs(int start, int end) {
    queue<int> q;
    q.push(start);
    v[start] = true;
    parent[start] = -1; // Start node has no parent

    while (!q.empty()) {
        int f = q.front();
        q.pop();

        if (f == end) break; // Stop if we reach the end node

        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!v[*i]) {
                q.push(*i);
                v[*i] = true;
                parent[*i] = f; // Set parent
            }
        }
    }

    vector<int> path;
    for (int i = end; i != -1; i = parent[i]) {
        path.push_back(i);
    }
    reverse(path.begin(), path.end()); // Reverse to get the correct order
    return path;
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

// Segment Tree
template<class Monoid> struct SegTree {
    using Func = function<Monoid(Monoid, Monoid)>;

    // core member
    int N;
    Func OP;
    Monoid IDENTITY;
    
    // inner data
    int offset;
    vector<Monoid> dat;

    // constructor
    SegTree() {}
    SegTree(int n, const Func &op, const Monoid &identity) {
        init(n, op, identity);
    }
    SegTree(const vector<Monoid> &v, const Func &op, const Monoid &identity) {
        init((int)v.size(), op, identity);
        build(v);
    }
    void init(int n, const Func &op, const Monoid &identity) {
        N = n;
        OP = op;
        IDENTITY = identity;
        offset = 1;
        while (offset < N) offset *= 2;
        dat.assign(offset * 2, IDENTITY);
    }
    void init(const vector<Monoid> &v, const Func &op, const Monoid &identity) {
        init((int)v.size(), op, identity);
        build(v);
    }
    void build(const vector<Monoid> &v) {
        assert(N == (int)v.size());
        for (int i = 0; i < N; ++i) dat[i + offset] = v[i];
        for (int k = offset - 1; k > 0; --k) dat[k] = OP(dat[k*2], dat[k*2+1]);
    }
    int size() const {
        return N;
    }
    Monoid operator [] (int a) const { return dat[a + offset]; }
    
    // update A[a], a is 0-indexed, O(log N)
    void set(int a, const Monoid &v) {
        int k = a + offset;
        dat[k] = v;
        while (k >>= 1) dat[k] = OP(dat[k*2], dat[k*2+1]);
    }
    
    // get [a, b), a and b are 0-indexed, O(log N)
    Monoid prod(int a, int b) {
        Monoid vleft = IDENTITY, vright = IDENTITY;
        for (int left = a + offset, right = b + offset; left < right;
        left >>= 1, right >>= 1) {
            if (left & 1) vleft = OP(vleft, dat[left++]);
            if (right & 1) vright = OP(dat[--right], vright);
        }
        return OP(vleft, vright);
    }
    Monoid all_prod() { return dat[1]; }
    
    // get max r that f(get(l, r)) = True (0-indexed), O(log N)
    // f(IDENTITY) need to be True
    int max_right(const function<bool(Monoid)> f, int l = 0) {
        if (l == N) return N;
        l += offset;
        Monoid sum = IDENTITY;
        do {
            while (l % 2 == 0) l >>= 1;
            if (!f(OP(sum, dat[l]))) {
                while (l < offset) {
                    l = l * 2;
                    if (f(OP(sum, dat[l]))) {
                        sum = OP(sum, dat[l]);
                        ++l;
                    }
                }
                return l - offset;
            }
            sum = OP(sum, dat[l]);
            ++l;
        } while ((l & -l) != l);  // stop if l = 2^e
        return N;
    }

    // get min l that f(get(l, r)) = True (0-indexed), O(log N)
    // f(IDENTITY) need to be True
    int min_left(const function<bool(Monoid)> f, int r = -1) {
        if (r == 0) return 0;
        if (r == -1) r = N;
        r += offset;
        Monoid sum = IDENTITY;
        do {
            --r;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!f(OP(dat[r], sum))) {
                while (r < offset) {
                    r = r * 2 + 1;
                    if (f(OP(dat[r], sum))) {
                        sum = OP(dat[r], sum);
                        --r;
                    }
                }
                return r + 1 - offset;
            }
            sum = OP(dat[r], sum);
        } while ((r & -r) != r);
        return 0;
    }
    
    // debug
    friend ostream& operator << (ostream &s, const SegTree &seg) {
        for (int i = 0; i < seg.size(); ++i) {
            s << seg[i];
            if (i != seg.size()-1) s << " ";
        }
        return s;
    }
};


#ifdef isym444_LOCAL
const string COLOR_RESET = "\033[0m", BRIGHT_GREEN = "\033[1;32m", BRIGHT_RED = "\033[1;31m", BRIGHT_CYAN = "\033[1;36m", NORMAL_CROSSED = "\033[0;9;37m", RED_BACKGROUND = "\033[1;41m", NORMAL_FAINT = "\033[0;2m";
#define dbg(x) std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << COLOR_RESET << std::endl
#define dbgif(cond, x) ((cond) ? std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << __FILE__ << COLOR_RESET << std::endl : std::cerr)
#else
#define dbg(x) ((void)0)
#define dbgif(cond, x) ((void)0)
#endif

#define rep(i,n) for (int i = 0; i < (n); ++i)


int op(int a, int b){
    return(max(a,b));
}

int e(){
    return 0;
}

int main(){
    int N,D;
    cin >> N >> D;
    vector<int> A(N);
    vector<int> B((int)5e5+5);
    cin >> A;

    // segtree<int, op, e> dp((int)5e5+5);
    SegTree<int> dp(B, [&](int a, int b){return max(a,b);}, 0);

    foi(0,N){
        int cur = A[i];
        int l = max(0,cur-D);
        int r = min((int)5e5+4,cur+D+1);
        int m = dp.prod(l,r)+1;
        dp.set(cur,m);
    }

    cout << dp.prod(0,(int)5e5+4) << endl;

    return 0;
}