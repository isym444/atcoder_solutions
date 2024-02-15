#include <iostream>
#include <set>
#define llint long long

using namespace std;

llint n, m;
llint a[5];

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0);
	
	cin >> n >> m;
	llint s, c;
	for(int i = 1; i <= n; i++) a[i] = -1;
	for(int i = 1; i <= m; i++){
		cin >> s >> c;
		if(a[s] != -1 && a[s] != c){
			cout << -1 << endl;
			return 0;
		}
		a[s] = c;
	}
	
	if(n == 1){
		if(a[1] <= 0){
			cout << 0 << endl;
			return 0;
		}
	}
	
	if(a[1] == 0){
		cout << -1 << endl;
		return 0;
	}
	if(a[1] == -1) a[1] = 1;
	for(int i = 2; i <= n; i++){
		if(a[i] == -1) a[i] = 0;
	}
	
	for(int i = 1; i<= n; i++)  cout << a[i]; cout << endl;
	
	return 0;
}