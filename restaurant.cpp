#include "main.h"	
bool comparePair(const std::pair<int, int>& i, const std::pair<int, int>& j) {
    if (i.second != j.second) {
        return i.second < j.second;
    } else {
        return i.first < j.first;
    }
}
vector<pair<char,int> > Freq_Appearance(string name)
{
	std::map<char,int> m;
	for(size_t i = 0; i<name.length();i++)
	{
		m[name[i]]++;
	}
	std::vector<std::pair<char,int> > v(m.begin(),m.end());
	std::sort(v.begin(),v.end(),comparePair);
	std::vector<std::pair<char,int> > v_upper;
    std::vector<std::pair<char,int> > v_lower;
    for(size_t i = 0; i < v.size(); i++){
        if(isupper(v[i].first))
            v_upper.push_back(make_pair(v[i].first,v[i].second));
        else v_lower.push_back(make_pair(v[i].first,v[i].second));
    }
    v.clear();

    for(size_t i = 0; i < v_lower.size(); i++){
        v.push_back(make_pair(v_lower[i].first, v_lower[i].second));
    }
    for(size_t i = 0; i < v_upper.size(); i++){
        v.push_back(make_pair(v_upper[i].first, v_upper[i].second));
    }
	return v;

}

void caesarCipher(vector<pair<char,int> > &name)
{
	size_t i;
	for(i = 0; i < name.size(); i++)
	{
		if(name[i].first >= 'a' && name[i].first <= 'z')
		{
			name[i].first = name[i].first + name[i].second;
			if(name[i].first > 'z')
			{
				name[i].first = name[i].first - 'z' + 'a' - 1;
			}
		}
		else if(name[i].first >= 'A' && name[i].first <= 'Z')
		{
			name[i].first = name[i].first + name[i].second;
			if(name[i].first > 'Z')
			{
				name[i].first = name[i].first - 'Z' + 'A' - 1;
			}
		}
	}
	
	std::map<char,int> freqMap;
	for(size_t j = 0; j < name.size(); j++)
	{
		freqMap[name[j].first]+=name[j].second;
	}
	name.clear();
	map<char,int>::iterator it;
	for(it = freqMap.begin(); it != freqMap.end(); it++)
	{
		name.push_back(make_pair(it->first,it->second));
	}
	std::sort(name.begin(),name.end(),comparePair);
	std::vector<std::pair<char,int> > v_upper;
    std::vector<std::pair<char,int> > v_lower;
    for(size_t i = 0; i < name.size(); i++){
        if(isupper(name[i].first))
            v_upper.push_back(make_pair(name[i].first,name[i].second));
        else v_lower.push_back(make_pair(name[i].first,name[i].second));
    }
    name.clear();

    for(size_t i = 0; i < v_lower.size(); i++){
        name.push_back(make_pair(v_lower[i].first, v_lower[i].second));
    }
    for(size_t i = 0; i < v_upper.size(); i++){
        name.push_back(make_pair(v_upper[i].first, v_upper[i].second));
    }
	return;
}
string transform_Caesar(std::string name) {
    std::string result = "";
    std::map<char,int> m;
    for(size_t i = 0; i<name.length();i++)
    {
        m[name[i]]++;
    }
    for (size_t i = 0; i < name.size(); i++) {
        char currentChar = name[i];

        if (isalpha(currentChar)) {
            char base = islower(currentChar) ? 'a' : 'A';
            char shiftedChar = static_cast<char>((currentChar - base + m[currentChar]) % 26 + base);
            result += shiftedChar;
        } else {
            result += currentChar; 
        }
    }

    return result;
}
template <typename T, typename Comp>
class Heap{
private:
	T* HeapArray;
	int size;
	int maxsize;
	void shiftdown(int pos){
		while(!isLeaf(pos)){
			int j = leftchild(pos);
			int rc = rightchild(pos);
			if((rc < size) && Comp::prior(HeapArray[rc], HeapArray[j]))
				j = rc;
			if(Comp::prior(HeapArray[pos], HeapArray[j]))
				return;
			swap(HeapArray, pos, j);
			pos = j;
		}
	}
public:
	Heap(T* h, int num, int max){
		HeapArray = h;
		size = num;
		maxsize = max;
		buildHeap();
	}
	int length() const {return size;}
	bool isLeaf(int pos) const {return (pos >= size/2) && (pos < size);}
	int leftchild(int pos) const {return 2*pos + 1;}
	int rightchild(int pos) const {return 2*pos + 2;}
	int parent(int pos) const {return (pos - 1)/2;}
	void buildHeap(){
		for(int i = size/2 - 1; i >= 0; i--)
			shiftdown(i);
	}
	void insert(const T& it){
		if(size >= maxsize)
			return;
		int curr = size++;
		HeapArray[curr] = it;
		while((curr != 0) && (Comp::prior(HeapArray[curr], HeapArray[parent(curr)]))){
			swap(HeapArray, curr, parent(curr));
			curr = parent(curr);
		}
	}
	T removefirst(){
		if(size == 0)
			return NULL;
		swap(HeapArray, 0, --size);
		if(size != 0)
			shiftdown(0);
		return HeapArray[size];
	}
	T remove(int pos){
		if(pos < 0 || pos >= size)
			return NULL;
		if(pos == size - 1)
			size--;
		else{
			swap(HeapArray, pos, --size);
			while((pos != 0) && (Comp::prior(HeapArray[pos], HeapArray[parent(pos)]))){
				swap(HeapArray, pos, parent(pos));
				pos = parent(pos);
			}
			if(size != 0)
				shiftdown(pos);
		}
		return HeapArray[size];
	}
};
template <typename T>
void swap(T* A, int i, int j){
	T temp = A[i];
	A[i] = A[j];
	A[j] = temp;
}
template <typename T, typename Comp>
class MinTree{
public:
	static bool prior(T a, T b){
		return Comp::prior(a, b);
	}
};
template <typename T>
class Compare{
public:
	static bool prior(T a, T b){
		return a->weight() < b->weight();
	}
};
template <typename T>
class HuffNode{
public:
    virtual ~HuffNode() {}
    virtual int weight() = 0;
    virtual bool isLeaf() = 0;
	virtual HuffNode<T>* left() const = 0;
    virtual HuffNode<T>* right() const = 0;
    virtual void setLeft(HuffNode<T>* b) = 0;
    virtual void setRight(HuffNode<T>* b) = 0;
	virtual int getH() = 0;
};
template <typename T>
class LeafNode : public HuffNode<T>{
private:
    T it; //value
    int wgt; //weight
public:
    LeafNode(const T& val, int freq) {it = val; wgt = freq;}
    int weight() {return wgt;}
    T val() {return it;}
    bool isLeaf() {return true;}
	HuffNode<T>* left() const { return NULL; }
    HuffNode<T>* right() const { return NULL; }
    void setLeft(HuffNode<T>* b) {}
    void setRight(HuffNode<T>* b) {}
	int getH() {return 0;}
};

template <typename T>
class IntlNode : public HuffNode<T>{
private:
    HuffNode<T>* lc; //left child
    HuffNode<T>* rc; //right child
    int wgt; //subtree weight
public:
    IntlNode(HuffNode<T>* l, HuffNode<T>* r) {wgt = l->weight() + r->weight(); lc = l; rc = r;}
    int weight() {return wgt;}
    bool isLeaf() {return false;}
    HuffNode<T>* left() const {return lc;}
    void setLeft(HuffNode<T>* b){
        lc = (HuffNode<T>*)b;
    }
    HuffNode<T>* right() const {return rc;}
    void setRight(HuffNode<T>* b){
        rc = (HuffNode<T>*)b;
    }
	int getH() {
		int leftH = 0;
		int rightH = 0;
		if(lc != NULL){
			leftH = lc->getH();
		}
		if(rc != NULL){
			rightH = rc->getH();
		}
		return 1 + max(leftH, rightH);
	
	}
	

};
template <typename T>
class HuffTree{
private:
    HuffNode<T>* Root; //tree root
public:
    HuffTree(T& val, int freq) {Root = new LeafNode<T>(val, freq);}
    //Internal node constructor
    HuffTree(HuffTree<T>* l, HuffTree<T>* r) {Root = new IntlNode<T>(l->root(),r->root());}
    ~HuffTree() {}
	void setRoot(HuffNode<T>* node) {Root = node;}
    HuffNode<T>* root() {return Root;}
    int weight() {return Root->weight();}
};
template <typename T>
int balanceFactor(HuffNode<T>* node){
	if(node->isLeaf()){
		return 0; //leaf has balance factor 0
	}
	else{
		return node->left()->getH() - node->right()->getH();
	}
}
template <typename T>
HuffNode<T>* rightRotation(HuffNode<T>* node){
	HuffNode<T>* temp = node->left();
	node->setLeft(temp->right());
	temp->setRight(node);
	node = temp;
	return node;
}
template <typename T>
HuffNode<T>* leftRotation(HuffNode<T>* node){
	HuffNode<T>* temp = node->right();
	node->setRight(temp->left());
	temp->setLeft(node);
	node = temp;
	return node;
}
template <typename T>
HuffNode<T>* balanceTree(HuffNode<T>* root){
	int balance = balanceFactor(root);
	if(balance>1){
		if(balanceFactor(root->left()) < 0){
			//Left-Right case
			root->setLeft(leftRotation(root->left()));
		}
		return rightRotation(root);
	}
	else if(balance<-1){
		//Right heavy
		if(balanceFactor(root->right()) > 0){
			//Right-Left case
			root->setRight(rightRotation(root->right()));
		}
		return leftRotation(root);
	}
	return root;
}

template <typename T>
HuffTree<T>* buildHuff(HuffTree<T>** TreeArray, int count){
	Heap<HuffTree<T>*, Compare<HuffTree<T>* > > minHeap(TreeArray, count, count);
	HuffTree<T>* temp1, *temp2, *temp3 = NULL;
	while(minHeap.length() > 1){
		int i = 3;
		temp1 = minHeap.removefirst();
		temp2 = minHeap.removefirst();
		temp3 = new HuffTree<T>(temp1, temp2);
		while(i!=0){
			temp3->setRoot(balanceTree(temp3->root()));
			i--;
		}
		if (temp3->root()->isLeaf()){
			return NULL;
		}
		minHeap.insert(temp3);
		delete temp1;
		delete temp2;
	}
	return temp3;
}

template <typename T>
void buildCode(HuffNode<T>* node, string code, std::map<char,string> &m){
	if(node->isLeaf()){
		LeafNode<T>* leaf = (LeafNode<T>*)node;
		m[leaf->val()] = code;
	}
	else{
		IntlNode<T>* intl = (IntlNode<T>*)node;
		buildCode(intl->left(), code + '0', m);
		buildCode(intl->right(), code + '1', m);
	}
}
template <typename T>
std::map<char,string> buildCode(HuffTree<T>* tree){
	std::map<char,string> m;
	buildCode(tree->root(), "", m);
	return m;
}
template <typename T>
string encode(HuffTree<T>* tree, string str, std::map<char,string> &m){
	std::map<char,string>::iterator it;
	string code = "";
	for(unsigned i = 0; i < str.length(); i++){
		it = m.find(str[i]);
		code+=it->second;
	}
	return code;
}
template <typename T>
string decode(HuffTree<T>* tree, string str){
	HuffNode<T>* curr = tree->root();
	string decoded = "";
	for(int i = 0; i < str.length(); i++){
		if(str[i] == '0'){
			curr = ((IntlNode<T>*)curr)->left();
		}
		else{
			curr = ((IntlNode<T>*)curr)->right();
		}
		if(curr->isLeaf()){
			decoded += ((LeafNode<T>*)curr)->val();
			curr = tree->root();
		}
	}
	return decoded;
}


string enCrypt(string name, int &Result,HuffTree<char>*& huffT)
{
	std::map<char,int> c;
	for(size_t i = 0; i<name.length();i++)
	{
		c[name[i]]++;
	}
	if(c.size() < 3){return "";}
	vector<pair<char,int> >v_name = Freq_Appearance(name);
	caesarCipher(v_name); 
	
	
	string new_name = transform_Caesar(name);
	
	HuffTree<char>** treeArr = new HuffTree<char>*[v_name.size()];
	for(size_t i = 0; i < v_name.size(); i++){
		treeArr[i] = new HuffTree<char>(v_name[i].first, v_name[i].second);
	}
	HuffTree<char>* huffmanTree = buildHuff(treeArr, v_name.size());
	huffT = huffmanTree;
	if(huffmanTree == NULL){ //root is a leaf
		
		return "";
	}
	map<char,string> m = buildCode(huffmanTree);
	string code = encode(huffmanTree, new_name, m);
	
	
	//take maximum 10 character in code string
	if(code.length() >= 10)
		code = code.substr(code.length()-10, 10);
	reverse(code.begin(), code.end());
	//transform binary string to decimal integer another way
	if(huffmanTree->root()->left() == NULL && huffmanTree->root()->right() == NULL){
		Result = 0;
	}
	else{
		for(unsigned i = 0; i < code.length(); i++){
		Result += (code[i] - '0') * pow(2, code.length() - i - 1);
		}	
	}
	
	return code;
}
class BST{
private:
	struct Node{
		int key;
		Node* left;
		Node* right;
		Node(int key){
			this->key = key;
			left = NULL;
			right = NULL;
		}
	};
	Node* root;
	int size;
	void insert(Node* &node, int key){
		if(node == NULL){
			node = new Node(key);
			size++;
		}
		else if(key < node->key){
			insert(node->left, key);
		}
		else if(key >= node->key){
			insert(node->right, key);
		}
	}
	bool find(Node* node, int key){
		if(node == NULL){
			return false;
		}
		else if(key < node->key){
			return find(node->left, key);
		}
		else if(key > node->key){
			return find(node->right, key);
		}
		else{
			return true;
		}
	}
	void remove(Node* &node, int key){
		if(node == NULL){
			return;
		}
		else if(key < node->key){
			remove(node->left, key);
		}
		else if(key > node->key){
			remove(node->right, key);
		}
		else{
			if(node->left == NULL && node->right == NULL){
				delete node;
				node = NULL;
				size--;
			}
			else if(node->left == NULL){
				Node* temp = node;
				node = node->right;
				delete temp;
				size--;
			}
			else if(node->right == NULL){
				Node* temp = node;
				node = node->left;
				delete temp;
				size--;
			}
			else{
				Node* temp = node->right;
				while(temp->left != NULL){
					temp = temp->left;
				}
				node->key = temp->key;
				remove(node->right, temp->key);
			}
		}
	}
	void print(Node* node){
		if(node == NULL){
			return;
		}
		print(node->left);
		cout<<node->key<<endl;
		print(node->right);
	}
public:
	BST(){
		root = NULL;
		size = 0;
	}
	~BST(){
		while(root != NULL){
			remove(root, root->key);
		}
	}
	void insert(int key){
		insert(root, key);
	}
	bool find(int key){
		return find(root, key);
	}
	void remove(int key){
		remove(root, key);
	}
	void clear(){
		while(root != NULL){
			remove(root, root->key);
		}
	}
	void print(){
		print(root);
	}
	int getSize(){
		return size;
	}
	void traverse_Post_Order(Node* root,vector<int> &v){
		if(root == NULL)
			return;
		traverse_Post_Order(root->left, v);
		traverse_Post_Order(root->right, v);
		v.push_back(root->key);
	}
	Node* getRoot() {return this->root;}
};
void calculateFact(int fact[], int N){
    fact[0] = 1;
    for(long long int i = 1; i < N; i++){
        fact[i] = fact[i-1]*i;
    }
}
int nCr(int fact[], int N, int R){
    if(R > N)
        return 0;
    int res = fact[N] / fact[R];
    res /= fact[N-R];

    return res;
}
int countWays(vector<int> &arr, int fact[]){
    int N = arr.size();

    if(N<=2)
        return 1;

    vector<int> leftSubTree;
    vector<int> rightSubTree;

    int root = arr[N-1];

    for(int i = 0; i < N -1; i++){
        if(arr[i] < root)
            leftSubTree.push_back(arr[i]);
        else
            rightSubTree.push_back(arr[i]);
    }

    int N1 = static_cast<int>(leftSubTree.size());
    //int N2 = static_cast<int>(rightSubTree.size());

    int countLeft = countWays(leftSubTree, fact);
    int countRight = countWays(rightSubTree, fact);

    return nCr(fact, N-1, N1)*countLeft*countRight;
}
class HashTable{ //hash table is like a matrix with hash code
public:
	int size;
	int capacity;
	BST** table;
	deque<int>** d; //FIFO
public:
	HashTable(int capacity){
		this->capacity = capacity+1;
		size = 0;
		table = new BST*[capacity + 1];
		d = new deque<int>*[capacity + 1];
		for(int i = 0; i < capacity + 1 ; i++){
			table[i] = new BST();
			d[i] = new deque<int>;
		}
	}
	~HashTable(){
		for(int i = 0; i < capacity ; i++){
			delete table[i];
			delete d[i];
		}
		delete[] table;
		delete[] d;
		
	}
	
	void insert(int key, int hashVal){
		table[hashVal]->insert(key);
		if (d[hashVal] == NULL) {d[hashVal] = new deque<int>;}
		d[hashVal]->push_back(key);
		updateSize();
	}
	bool find(int key, int hashVal){
		return table[hashVal]->find(key);
	}
	void remove(){
		if(size==0)
			return;
		for(int i = 1; i < capacity ; i++){
			if(table[i]->getSize() == 0)
				continue;
			vector<int> v;
			table[i]->traverse_Post_Order(table[i]->getRoot(),v);
			//Number of permutations generating the same BST from vector v
			int N = v.size();

			int* fact = new int[N];

			calculateFact(fact, N);

			int count = countWays(v, fact);

			count %= capacity;
			if(count >= table[i]->getSize())
				table[i]->clear();
			for(int j = 0; j < count; j++){
				int key = d[i]->front();
				d[i]->pop_front();
				table[i]->remove(key);
			}
			delete[] fact;
		}
		updateSize();
		
	}
	int getSize(){
		return size;
	}
	void updateSize(){
		int count = 0;
		for(int i = 0; i < capacity; i++){
			if(table[i]->getSize()>0){
				count++;
			}
		}
		size = count;
	}
	int getCapacity(){
		return capacity;
	}
	void print(){
		for(int i = 0; i < capacity; i++){
			table[i]->print();
		}
	}
};
unsigned findIndex_Deque(const deque<int> &d, int key){
	for(unsigned i = 0; i < d.size(); i++){
		if(d[i] == key){
			return i;
		}
	}
	return -1;
}
class Heap_R{
public:
	pair<int,int>* heap; //<id, key>
	int size;
	int capacity;
	deque<int>** d; //FIFO
	deque<int> priority;
	
	void shiftdown(int pos){
		while(!isLeaf(pos)){
			int j = leftchild(pos);
			int rc = rightchild(pos);
			if((rc < size)){
				if(heap[rc].second == heap[j].second){
					if(findIndex_Deque(priority,heap[rc].first) > findIndex_Deque(priority,heap[j].first)){
						j = rc;
					}	
				}
				if(heap[rc].second < heap[j].second)
					j = rc;
			}
			if(heap[pos].second < heap[j].second)
				return;
			swap(pos, j);
			pos = j;
		}
	}
public:
	Heap_R(int capacity){
		this->capacity = capacity;//1 to MAXSIZE
		size = 0;
		d = new deque<int>*[capacity];
		heap = new pair<int,int>[capacity];
		for(int i = 0; i < capacity; i++){
			heap[i].first = i+1;
			heap[i].second = INT_MAX;
			d[i] = new deque<int>;
		}
	}
	~Heap_R(){
		delete[] heap;
		for(int i = 0; i < capacity; i++){
			delete d[i];
		}
		delete[] d;
	}
	bool isLeaf(int pos) const {return (pos >= capacity/2) && (pos < capacity);}
	int leftchild(int pos) const {return 2*pos + 1;}
	int rightchild(int pos) const {return 2*pos + 2;}
	int parent(int pos) const {return (pos - 1)/2;}
	int getSize(){ //acctually size
		int count = 0;
		for(int i = 0; i < capacity; i++){
			if(heap[i].second!=INT_MAX){
				count++;
			}
		}
		return count;
	}
	void updateSize(){
		size = getSize();
	}
	void insert(int id, int key){
		if(size >= capacity)
			return;
		int curr;
		if (d[id] == NULL) {d[id] = new deque<int>;}
		d[id]->push_back(key);
		priority.push_front(id);
		for(int i = 0; i  < capacity; i++){
			if(heap[i].first == id){
				curr = i;
			}
		}
		if(heap[curr].second == INT_MAX)
			heap[curr].second = 0;
		heap[curr].second++;
		updateSize();
		while((curr != 0) && (heap[curr].second < heap[parent(curr)].second)){ //reheap up
			swap(curr, parent(curr));
			curr = parent(curr);
		}
		if((rightchild(curr)<capacity || leftchild(curr)<capacity) && ((heap[curr].second >= heap[leftchild(curr)].second) || 
		(heap[curr].second >= heap[rightchild(curr)].second))) //reheap down
			shiftdown(curr);
		priority.push_front(heap[curr].first);

	}
	void remove(int num){ 
		if(this->getSize() == 0)
			return;
		int n = num;
		vector<pair<int, int> > v; //to print
		vector<bool> check_remove = vector<bool>(capacity+1, false); //index tính theo id
		while(n>0){
			int min = heap[0].second;
			
			int id = 0; //index
			
			for(int i = 1;i<capacity;i++){
				if(heap[i].second != INT_MAX && heap[i].second < min && check_remove[heap[i].first] == false){
					min = heap[i].second;
					id = i;
					
				}
			}
			
			for(int j = 0; j < capacity; j++){
				if(heap[j].second == min && findIndex_Deque(priority,heap[j].first) > findIndex_Deque(priority,heap[id].first)) {
					id = j;
				}
			}
			
			if(num>heap[id].second){
				heap[id].second = INT_MAX;
				for(unsigned i = 0; i < d[id]->size(); i++){
					v.push_back(make_pair(heap[id].first, d[heap[id].first]->front()));
					d[heap[id].first]->pop_front();
				}
				
			}
			else{
				heap[id].second -= num;
				for(int j = 0; j < num; j++){
					v.push_back(make_pair(heap[id].first, d[heap[id].first]->front()));
					d[heap[id].first]->pop_front();
				}
			}
			
			
    		check_remove[heap[id].first] = true;
			
			priority.push_front(heap[id].first);
			n--;
			updateSize();
			for(int i = this->getSize()/2 -1 ; i >= 0; i--) //rebuild heap
				shiftdown(i);
		}
		
		for(unsigned i = 0; i < v.size(); i++){
			cout<<v[i].second<<"-"<<v[i].first<<endl;
		}
	}
	
	void swap(int i, int j){
		int temp1 = heap[i].first;
		heap[i].first = heap[j].first;
		heap[j].first = temp1;

		int temp2 = heap[i].second;
		heap[i].second = heap[j].second;
		heap[j].second = temp2;
	}
	void pre_Order(int index, int num){
		if (heap[index].second != INT_MAX) {
            // Process the current node
			if(heap[index].first < capacity+1){
            if(num >= static_cast<int>(d[heap[index].first]->size())){
				for(int i = d[heap[index].first]->size() - 1; i>=0 ; i--){
					cout<<heap[index].first<<"-"<<d[heap[index].first]->at(i)<<endl;
					
				}
				
			}
			else{
				int k = num;
				
				size_t i = d[heap[index].first]->size() - 1;
				while(k>0){
					cout<<heap[index].first<<"-"<<d[heap[index].first]->at(i)<<endl;
					
					k--;
					i--;
				}
			}

            // Recursively traverse the left and right subtrees
            pre_Order(2 * index + 1, num); // Left child
            pre_Order(2 * index + 2, num); // Right child
			}
			
        }
	}
};

void LAPSE(int Result, int MAXSIZE, HashTable *G_Rest, Heap_R *S_Rest){ 
	int ID = Result % MAXSIZE + 1;
	if(Result%2!=0) {
		G_Rest->insert(Result, ID);
	}
	else{
		S_Rest->insert(ID, Result);
	}
}
void KOKUSEN(HashTable *G_Rest){
	if(G_Rest->getSize() == 0){
		return;
	}
	else{
		G_Rest->remove();
	}
}
void KEITEIKEN(Heap_R *S_Rest, int num){
	if(S_Rest->getSize() == 0){
		return;
	}
	else{
		S_Rest->remove(num);
	}
}
void inOrder_hand(HuffNode<char>* node){
	if(node == NULL){
		return;
	}
	inOrder_hand(node->left());
	if(node->isLeaf()){
		cout<<((LeafNode<char>*)node)->val()<<endl;
	}
	else{
		cout<<node->weight()<<endl;
	}
	
	inOrder_hand(node->right());
}
void HAND(HuffTree<char>* huffT){
	if(huffT == NULL){
		return;
	}
	inOrder_hand(huffT->root());
}
void LIMITLESS(HashTable *G_Rest, int num){
	if(num > G_Rest->getCapacity() || num < 1){
		return;
	}
	if(G_Rest->table[num]->getSize() == 0){
		return;
	}
	G_Rest->table[num]->print();
}
void CLEAVE(Heap_R *S_rest, int num){
	if(S_rest->getSize() == 0){
		return;
	}
	S_rest->pre_Order(0, num);
}
void simulate(string filename)
{
	ifstream ss(filename.c_str());
	int MAXSIZE, num;
	string str, maxsize, name, n;
	int Result;
	HashTable* Gojo_Restaurant = NULL;
	Heap_R* Sukuna_Restaurant = NULL;
	HuffTree<char>* huffT = NULL; //to in Hand
	while(ss >> str)
	{ 
		if(str == "MAXSIZE")
		{
			ss >> maxsize;
			stringstream ss(maxsize);
			ss >> MAXSIZE ; 
			//cout<<MAXSIZE<<endl;
			Gojo_Restaurant = new HashTable(MAXSIZE);
			Sukuna_Restaurant = new Heap_R(MAXSIZE);
    	}
        else if(str == "LAPSE") // LAPSE <NAME>
        {
            ss >> name;
            //cout<<name<<endl;
			Result = 0;
			string code = enCrypt(name, Result, huffT);
			if(code.empty()){
				continue;
			}
			LAPSE(Result,MAXSIZE,Gojo_Restaurant,Sukuna_Restaurant);
			//cout<<code<<endl;
			//cout<<Result<<endl;
			
    	}
		else if(str == "KOKUSEN"){
			KOKUSEN(Gojo_Restaurant);
		}
		else if(str == "KEITEIKEN"){
			ss >> n;
			stringstream ss(n);
			ss >> num;
			KEITEIKEN(Sukuna_Restaurant, num);
		}
		else if(str == "HAND"){
			HAND(huffT);
		}
		else if(str == "LIMITLESS"){
			ss >> n;
			stringstream ss(n);
			ss >> num;
			LIMITLESS(Gojo_Restaurant, num);
		}
		else if(str == "CLEAVE"){
			ss >> n;
			stringstream ss(n);
			ss >> num;
			CLEAVE(Sukuna_Restaurant, num);
		}
		
    }
	delete Gojo_Restaurant;
	delete Sukuna_Restaurant;
	
}