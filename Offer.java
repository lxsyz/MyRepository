import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Stack;


import MirrorTree.Soluction.ListNode;

public class Offer {

	public static void main(String[] args) {
    	Offer offer = new Offer();
    	
    	int[] arr = {2,3,4,2,6,2,5,1};
//    	offer.heapOp(arr, 4);
//    	offer.findNums(arr, new int[1], new int[1]);
//    	char[] cs = {'b', 'c','b', 'b', 'a', 'b', 'a', 'b'};
//    	char[] p = {'.', '*', 'a', '*', 'a'};
//    	offer.match(cs, p);
    }
	
	/*
	 * 判断B树是不是A树的子树
	 */
	public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        boolean res = false;
        
        if (root1 != null && root2 != null) {
            if (root1.val == root2.val) {
                res = subTree(root1, root2);
            }
            
            if (!res) {
                res = HasSubtree(root1.left, root2);
            }
            
            if (!res) {
                res = HasSubtree(root1.right, root2);
            }
        }
        return res;
    }
    
    public boolean subTree(TreeNode rootA, TreeNode rootB) {
        if (rootB == null)
            return true;
        if (rootA == null)
            return false;
        if (rootA.val != rootB.val) {
            return false;
        }    
        
        if (rootA.val == rootB.val) {
              return subTree(rootA.left, rootB.left) && 
                  subTree(rootA.right, rootB.right);
        }    
        return false;
    }
    
    Stack<Integer> stack = new Stack<Integer>();    
    Stack<Integer> extraStack = new Stack<Integer>();
    public void push(int node) {
        stack.push(node);
        if (extraStack.isEmpty()) {
        	extraStack.push(node);
        } else {
        	if (node < extraStack.peek()) {
        		extraStack.push(node);
        	} else {
        		extraStack.push(extraStack.peek());
        	}
        }
    }
    
    public void pop() {
        stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        
        return extraStack.pop();
    }
    
    /*
     * 二叉搜索树的后序遍历序列
     * 1. i从左往右扫，遇到大于根节点的值，停下，i左边的为左子树
     * 2. j从右往左扫，遇到小于根节点的值，返回false，j经过的为右子树
     * 3. 递归的扫描左子树和右子树
     */
    public boolean VerifySquenceOfBST(int [] sequence) {
        if (sequence.length == 0)
            return false;
        return judge(sequence,0,sequence.length - 1);
    }
    
    public boolean judge(int[] arr,int start,int end) {
        if (end <= start) return true;
        int i = start;
        for (;i < end;i++)
            if (arr[i] > arr[end])
            	break;
        for (int j = i;j < end;j++) 
            if (arr[j] < arr[end])
            	return false;
		
        return judge(arr,start,i-1) && judge(arr,i,end - 1);
        
    }
    
    /**
     * 打印树的节点和为指定值的路径
	 * 从根节点扫描树，如果为叶子节点且和为target就add，注意这里的add要add一个new对象，否则修改path也会修改最终结果
	 * 小于值就扫描左右子树
     */
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        int res = 0;
        ArrayList<Integer> path = new ArrayList<Integer>();
        ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
        if (root!= null)
            find(list, path, root, res, target);
        return list;
    }
    
    public void find(ArrayList<ArrayList<Integer>> list, ArrayList<Integer> path, TreeNode node, int res, int target) {
        int val = node.val;
        res += val;
        path.add(val);

        if (res < target) {
            if (node.left != null) {
                find(list, path, node.left, res, target);
            }
            if (node.right != null) {
                find(list, path, node.right, res, target);
            }
        } else if (res == target && node.left == null && node.right == null) {
            list.add(path);
            //return;
        }
        
        path.remove(path.size()-1);
    }


    /**
     * 复杂链表的复制   哈希表解法
     * 哈希表存储原始链表节点与新链表节点的对应关系，在插入random时进行查找
     * @param pHead
     * @return
     */
    public RandomListNode Clone(RandomListNode pHead)
    {
        if (pHead == null)
            return null;
        
        RandomListNode newHead = new RandomListNode(pHead.label);
        HashMap<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode temp = newHead;
        RandomListNode temp2 = pHead.next;
        while (temp2 != null) {
            RandomListNode node = new RandomListNode(temp2.label);
            temp.next = node;
            temp = node;
            
            map.put(temp2, temp);

            temp2 = temp2.next;
            
        }
        
        RandomListNode temp3 = newHead;
        RandomListNode temp4 = pHead;
        while (temp4 != null) {
            if (temp4.random != null) {
                temp3.random = map.get(temp4.random);
            }
            temp4 = temp4.next;
            temp3 = temp3.next;
        }
        return newHead;
    }
    
    /**
     * 复杂链表复制    O1空间解法
     * @author Administrator
     *
     */
    public RandomListNode Clone2(RandomListNode pHead) {
    	if(pHead==null)
            return null;
        RandomListNode pCur = pHead;
        //复制next 如原来是A->B->C 变成A->A'->B->B'->C->C'
        while(pCur!=null){
            RandomListNode node = new RandomListNode(pCur.label);
            node.next = pCur.next;
            pCur.next = node;
            pCur = node.next;
        }
        pCur = pHead;
        //复制random pCur是原来链表的结点 pCur.next是复制pCur的结点
        while(pCur!=null){
            if(pCur.random!=null)
                pCur.next.random = pCur.random.next;
            pCur = pCur.next.next;
        }
        RandomListNode head = pHead.next;
        RandomListNode cur = head;
        pCur = pHead;
        //拆分链表
        while(pCur!=null){
            pCur.next = pCur.next.next;
            if(cur.next!=null)
                cur.next = cur.next.next;
            cur = cur.next;
            pCur = pCur.next;
        }
        return head;       
    }

 
	public TreeNode Convert(TreeNode root) {
		if (root == null)
			return null;

		if (root.left == null && root.right == null)
			return root;

		// 1.将左子树构造成双链表，并返回链表头节点
		TreeNode left = Convert(root.left);
		TreeNode p = left;
		// 2.定位至左子树双链表最后一个节点
		while (p != null && p.right != null) {
			p = p.right;
		}
		// 3.如果左子树链表不为空的话，将当前root追加到左子树链表
		if (left != null) {
			p.right = root;
			root.left = p;
		}
		// 4.将右子树构造成双链表，并返回链表头节点
		TreeNode right = Convert(root.right);
		// 5.如果右子树链表不为空的话，将该链表追加到root节点之后
		if (right != null) {
			right.left = root;
			root.right = right;
		}
		return left != null ? left : root;
	}
        
	public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> list = new ArrayList<>();
        if (input.length == 0)
            return null;
        if (k > input.length || k < 0)
            return null;
        int begin = 0;
        int end = input.length - 1;
        
        int index = partition(input, 0, input.length - 1);
        while (index != k - 1) {
            if (index > k - 1) {
                end = index - 1;
                index = partition(input, begin, end);
            } else if (index < k - 1) {
                begin = index + 1;
                index = partition(input, begin, end);
            }
        }
        for (int i = 0;i < k;i++)
            list.add(input[i]);
        return list;
    }
	
	/*
	 * 堆排序
	 * 选出最小的k个数
	 */
	public void heapOp(int[] input, int k) {
		for (int i = 0;i < k;i++) {
            heapSort(input, i, input.length);
        }
	}
	
	public void heapSort(int[] arr, int i, int len) {
        for (int j = len - 1;j >= i;j--) {
            int p = (j + i - 1) / 2;
            if (arr[p] > arr[j]) {
                swap(arr, p, j);
            }
        }
    }
	
	public int partition(int[] arr, int start, int end) {
		
		int small = start - 1;
		swap(arr, end, start);
		for (int i = start;i <= end;i++) {
			if (arr[i] < arr[end]) {
				small++;
				if (small != i) {
					swap(arr, small, i);
				}
			}
		}
		++small;
		swap(arr, small, end);
		return small;
	}

	public void quickSort(int[] arr, int start, int end) {
		if (start == end)
			return;
		int index = partition(arr, start, end);
		if (index > start)
			quickSort(arr, start, index - 1);
		if (index < end)
			quickSort(arr, index + 1, end);
			
	}
    
    
    /* 
     * 判断平衡二叉树
     */
	public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null)
            return true;
        
        return TreeDepth(root) == -1?false:true;
    }
    
    /*
     * 后序遍历判断每个节点
     */
    public int TreeDepth(TreeNode root) {
        if (root == null)
            return 0;
        int left = TreeDepth(root.left);
        int right = TreeDepth(root.right);
        if (left == -1 || right == -1)
            return -1;
        
        int diff = left - right;
        
        if (diff > 1 || diff < -1)
            return -1;
        
        return left > right ? left +1 :right + 1;
    }
    
    /*
     * 查找数组中两个只出现了一次的数字,其他数字都出现了两次
     */
    public void findNums(int[] array, int num1[], int num2[]) {
    	if (array == null || array.length <= 0)
    		return;
    	int sum = 0;
    	// 做异或运算
    	for (int i = 0;i < array.length;i++) {
    		sum ^= array[i];
    	}
    	int flag = 1;
    	// 异或的结果中必然有1，也是剩余的两个数不相同的位置
    	// 所以可以根据异或结果中 1 的位置将原数组分为两组
    	// 寻找异或的结果中第一个为1的二进制位
    	// flag 左移
    	while ((sum & flag) == 0)
    		flag <<= 1;
    	
    	// 根据那一个位置是否有1，将原数组分为两组
    	for (int i = 0;i < array.length;i++) {
    		if ((array[i] & flag) == 0)
    			num1[0] ^= array[i];
    		else
    			num2[0] ^= array[i];
    	}
    }
    
    /*
     * 找到连续的序列，和为k
     * 从1加到n，和为n(n+1)/2 = S, 则n < 根号2S
     * 当序列长度n为偶数时，那么序列中间两个数的平均值是中间值， (sum % n)* 2 = n
     * 当序列长度n为奇数时,那么序列中间的数就是中间值， n & 1==1 && sum % n == 0
     * 
     */
    public ArrayList<ArrayList<Integer>> findContinuousSequence(int[] array, int k) {
    	ArrayList<ArrayList<Integer>> list = new ArrayList<>();
    	for (int n = (int)Math.sqrt(k * 2);n >= 1;n--) {
    		if ((n & 1) == 1 && k % n == 0 || (k % n) * 2 == n) {
    			ArrayList<Integer> temp = new ArrayList<>();
    			for (int j = 0, m = k/n - (n - 1) / 2;j < n;j++,k++)
    				temp.add(m);
    			list.add(temp);
    		}
    	}
    	return list;
    }
    
    public boolean isContinueous(int[] numbers) {
    	if (numbers.length != 5)
    		return false;
    	int[] a = new int[14];
    	for (int i = 0;i < numbers.length;i++) {
    		a[numbers[i]] += 1;
    	}
    	int interval = 0;
    	int last = 0;
    	for (int i = 0;i < a.length;i++) {
    		// 有相等的非零数
    		if (i != 0 && a[i] >=2) {
    			return false;
    		}
    		// 1 3 5等非零数的间隔长度
    		if (i != 0 && a[i] == 1) {
    			if (last != 0) {
					interval = interval + i - last - 1;
				}
    			last = i;
    		}
    	}
    	
    	if (interval > a[0]) {
    		return false;
    	}
    	return true;
    }
    
    /**
     * 约瑟夫环，当m-1出队后，剩余的元素为 m,m+1,...,0,1,...m-2
     * 将剩下的元素再作为新元素
     * @param n
     * @param m
     * @return
     */
    public int lastRemaining(int n, int m) {
    	if (n == 1)
    		return 0;
    	else{
    		return lastRemaining(n-1, m) % n;
    	}
    	
//    	int result = 0;
//    	for (int i = 2;i < n;i++) {
//    		result = (result + m) % i;
//    	}
//    	return result;
    }
    
    
    /**
     * 不用加减乘除做加法
     * 加法相当于异或运算，进位考虑与运算,当进位中没有1时，结束
     * @param num1
     * @param num2
     * @return
     */
    public int Add(int num1, int num2) {
    	int temp2 = 1;
    	while (temp2 != 0) {
    		int temp1 = num1 ^ num2;
    		temp2 = num1 & num2;
    		num1 = temp1;
    		num2 = temp2 << 1;
    	}
    	return num1;
    }
    
    /**
     * 字符串转int类型，考虑如下几个方面
     * 正负号，溢出
     * @param str
     * @return
     */
    public static boolean flag;
    public int strToInt(String str) {
    	flag = false;
    	if (str == null || str.trim().equals("")) {
    		flag =true;
    		return 0;
    	}
    	int symbol = 0;
    	int start = 0;
    	char[] arr = str.toCharArray();
    	if (arr[0] == '+') {
    		start++;
    		symbol = 1;
    	} else if (arr[0] == '-') {
    		start++;
    		symbol = 0;
    	}
    	int result = 0;
    	for (int i = start;i < arr.length;i++) {
    		if (arr[i] > '9' || arr[i] < '0') {
    			flag = true;
    			return 0;
    		}
    		int sum = result * 10 + (int)(arr[i] - '0');
    		
    		if ((sum - (int)(arr[i] - '0')) / 10 != result) {
    			flag = true;
    			return 0;
    		}
    		result = result * 10 + (int)(arr[i] - '0');
    	}
    	
    	result = (int)Math.pow(-1, symbol) * result;
    	return result;
    }
    
    
    public void swap(int[] arr, int i, int j) {
    	int temp = arr[i];
    	arr[i] = arr[j];
    	arr[j] = temp;
    }
    
    /**
     * 给定一个数组A[0,1,...,n-1],
     * 数组B[0,1,...,n-1],
     * 其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。
     * 不能使用除法
     * @author Administrator
     * 可以看作矩阵
     * [ 1， A1，A2，A3....An-1]
     * [A0, 1, A2...     An-1]
     * [A0, A1, 1.......An-1]
     * 每一行乘积就是B中一个元素
     */
    public int[] multiply(int[] A) {
    	int length = A.length;
		int[] B = new int[length];
		if (length != 0) {
			//先做下三角的乘积
			B[0] = 1;
			for (int i = 1;i < length;i++) {
				B[i] = B[i - 1] * A[i-1];
			}
			int temp = 1;
			// 倒着做上三角的乘积
			for (int j = length - 1;j >= 0;j--) {
				B[j] = B[j] * temp;
				temp = temp * A[j];
			}
		}
		return B;
	}
    
    public boolean match(char[] str, char[] pattern) {
    	if (str == null || pattern == null) {
    		return false;
    	}
    	int strIndex = 0;
    	int patternIndex = 0;
    	return matchCore(str, strIndex, pattern, patternIndex);
    }
    
    public boolean matchCore(char[] str, int strIndex, char[] pattern, int patternIndex) {
		if (strIndex == str.length && patternIndex == pattern.length)
			return true;
		// pattern先到尾部，匹配失败
		if (strIndex !=str.length && patternIndex == pattern.length)
			return false;
		// 模式第二个是*, 且字符串第一个跟模式第一个匹配，分3种匹配模式，如不匹配，模式后移两位
		if (patternIndex + 1 < pattern.length  && pattern[patternIndex + 1] == '*') {
			if ((strIndex != str.length && pattern[patternIndex] == str[strIndex])
					|| (strIndex != str.length && pattern[patternIndex] == '.')) {
				return matchCore(str, strIndex, pattern, patternIndex + 2) //模式后移2， 视为x*匹配0个字符
//						|| matchCore(str, strIndex+1, pattern, patternIndex + 2) // 模式匹配了一个字符
						|| matchCore(str, strIndex+1, pattern, patternIndex); // *匹配一个，再匹配后面的
			} else {
				return matchCore(str, strIndex, pattern, patternIndex + 2);
			}
			//模式第2个不是*，且字符串第1个跟模式第1个匹配，则都后移1位，否则直接返回false
		} else if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) || (pattern[patternIndex] == '.' && strIndex != str.length)) {
	        return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
	    }
	    return false;
		
	}	

    /**
     * 找到环的入口节点
     * @author Administrator
     * 设慢指针走了s，则s + nr = 2s
     * 相遇点到环入口处距离为y，起点到环入口处距离为x
     * 则 s = x + y
     * 则 nr = x + y, x = nr - y
     * nr - y就是慢指针在相遇处开始走的时间消耗
     *
     */
    public RandomListNode EntryNodeOfLoop(RandomListNode pHead) {
    	RandomListNode pSlow = pHead;
    	RandomListNode pFast = pHead;
    	boolean hasCircle = false;
    	while (pSlow != null && pFast.next != null) {
    		pSlow = pSlow.next;
    		pFast = pFast.next;
    		if (pFast.next != null) {
    			pFast = pFast.next;
    		}
    		
    		if (pFast == pSlow) {
    			hasCircle = true;
    			break;
    		}
    	}
    	if (!hasCircle)
    		return null;
    	
    	
    	pFast = pHead;
    	while (pSlow != pFast) {
    		pSlow = pSlow.next;
    		pFast = pFast.next;
    	}
    	return pSlow;
    }

    /**
     * 删除链表中重复节点
     * 考虑  1-1-1-1-1-2，此时需要注意声明一个表头，返回表头的next
     * 保存当前指针的值，如果后节点的值等于它，current指针一直后移，最后prev.next指向current
     * 如果后节点的值不等于它，prev = current, current后移
     * @author Administrator
     *
     */
    public ListNode deleteDuplication(ListNode pHead) {
    	if (pHead == null)
    		return null;
    	ListNode pFirst = new ListNode(-1);
    	ListNode pCur = pHead;
    	ListNode pPrev = pFirst;
    	
    	while (pCur != null && pCur.next != null) {
    		if (pCur.val == pCur.next.val) {
    			int val = pCur.val;
    			while (pCur != null && pCur.val == val)
    				pCur = pCur.next;
    			pPrev.next = pCur;
    		} else {
    			pPrev = pCur;
    			pCur = pCur.next;
    		}
    	}
    	return pFirst.next;
    }
    
    /**
     * 中序遍历顺序的下一个结点并且返回
     * 三种情况：
     * 1. 根为空
     * 2. 查找的节点有右孩子节点，寻找右孩子节点的左孩子，直到找到叶子节点，返回，如果没有左孩子，返回右孩子节点
     * 3. 查找节点是父节点的左孩子，返回父节点；如果不是，向上遍历父节点，并重复判断是否是父节点左孩子。
     * @return
     */
    public TreeNode getNextTreeNode(TreeNode pNode) {
    	if (pNode == null)
    		return null;
    	if (pNode.right != null) {
    		TreeNode temp = pNode.right;
    		while (temp.left != null) {
    			temp = temp.left;
    		}
    		return temp;
    	} else {
    		while (pNode.next != null) {
    			if (pNode.next.left == pNode) {
    				return pNode.next;
    			} else {
    				pNode = pNode.next;
    			}
    		}
    		return pNode.next;
    	}
    }
    
    public ArrayList<ArrayList<Integer>> printTreeNode(TreeNode pRoot) {
    	int layer = 1;
    	// 存奇数层节点
    	Stack<TreeNode> s1 = new Stack<>();
    	s1.push(pRoot);
    	Stack<TreeNode> s2 = new Stack<>();
    	ArrayList<ArrayList<Integer>> res = new ArrayList<>();
    	
    	while (!s1.empty() || !s2.empty()) {
    		if (layer % 2 != 0) {
    			ArrayList<Integer> temp = new ArrayList<>();
    			while (!s1.empty()) {
    				TreeNode node = s1.pop();
    				if (node != null) {
    					temp.add(node.val);
    					// 下一层存入s2中
    					s2.push(node.left);
    					s2.push(node.right);
    				}
    			}
    			if (!temp.isEmpty()) {
    				res.add(temp);
    				layer++;
    			}
    		} else {
    			// 偶数层
    			ArrayList<Integer> temp = new ArrayList<>();
    			while (!s2.empty()) {
    				TreeNode node = s2.pop();
    				if (node != null) {
    					temp.add(node.val);
    					// 下一层存入s1中，先存右节点
    					s1.push(node.right);
    					s1.push(node.left);
    				}
    			}
    			
    			if (!temp.isEmpty()) {
    				res.add(temp);
    				layer++;
    			}
    		}
    	}
    	return res;
    }
    
    /**
     * 广度优先遍历
     * @param pRoot
     * @return
     */
    public ArrayList<ArrayList<Integer>> printTree(TreeNode pRoot) {
    	LinkedList<TreeNode> queue = new LinkedList<>();
    	ArrayList<ArrayList<Integer>> list = new ArrayList<>();
    	ArrayList<Integer> temp = new ArrayList<>();
    	if (pRoot == null)
    		return list;
    	queue.addLast(pRoot);
    	queue.addLast(null);
    	while (!queue.isEmpty()) {
    		TreeNode node = queue.removeFirst();
    		if (node != null) {
    			temp.add(node.val);
    		} else {
    			list.add(new ArrayList<>(temp));
    			// O(n)遍历一遍list
    			temp.clear();
    			if (!queue.isEmpty()) {
    				queue.addLast(null);
    			} else
    				break;
    			continue;
    		}
    		
    		if (node.left != null)
    			queue.addLast(node.left);
    		if (node.right != null)
    			queue.addLast(node.right);
    		// 层分隔符
    	}
    	return list;
    }
    
    /** 
     * 
     * 获取中位数的数据流
     * 1. 为了保证数据平均分配，可以在数据的总数目是偶数时，把新数据插入到最小堆，数目为奇数时，新数据插入到最大堆
     * 2. 还要保证最大堆中所有元素都要小于最小堆中数据
     * 当数据要插入到最小堆时，先把它插入到最大堆，接着把最大堆中堆顶拿出来插入最小堆，这样最小堆中数据都大于最大堆
     * @author Administrator
     */
    private PriorityQueue<Integer> minQueue = new PriorityQueue<>();
	private PriorityQueue<Integer> maxQueue = new PriorityQueue<>(15, new Comparator<Integer>() {

		@Override
		public int compare(Integer o1, Integer o2) {
			return o2-o1;
		}
		
	});
	private int count = 0;
    public void insertHeap(Integer num) {
    	if ( count % 2 == 0) {
    		maxQueue.offer(num);
    		int temp = maxQueue.poll();
    		minQueue.offer(temp);
    	} else {
    		minQueue.offer(num);
    		int temp = minQueue.poll();
    		maxQueue.offer(temp);
    	}
    }
    
    /**
     * 考虑数据 2-3-4-2-6-2-5-1
     * 1. 队列为空，位置索引入队
     * 2. 用begin表示滑窗的起始位置，当begin>队列中第一个元素位置时，删除第一个元素
     * 3. 当新元素大于队列中元素时，删除队尾元素
     * 4. 当begin大于0时
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxWindow(int[] num, int size) {
    	ArrayList<Integer> res = new ArrayList<>();
    	if (size <= 0)
    		return res;
    	ArrayDeque<Integer> q = new ArrayDeque<>();
    	int begin;
    	for (int i = 0;i < num.length;i++) {
    		begin = i - size + 1;
    		//如果队列为空，位置索引 入队
    		if (!q.isEmpty()) {
    			// 如果滑窗的起始位置大于队列的第一个元素了，移除队列的第一个元素
    			if (begin > q.peekFirst()) {
    				q.pollFirst();
    			}
    		}
    		// 当队列不为空时，新元素比队列中后面元素大，那么移除后面元素
    		// 这样队列中第一个元素一定最大
    		while (!q.isEmpty() && num[q.peekLast()] < num[i]) {
    			q.removeLast();
    		}
    		q.add(i);
    		if (begin >= 0) {
    			res.add(num[q.peekFirst()]);
    		}
    	}
    	return res;
    }
    
    boolean isSymmetrical(TreeNode pRoot)
    {
        if (pRoot == null)
            return true;
        return func(pRoot.left, pRoot.right);
    }
    
    boolean func(TreeNode left, TreeNode right) {
        if (left == null) {
            return right == null;
        }
        if (right == null)
            return false;
        if (left.val != right.val) return false;
        // 左子树的右子树和右子树的左子树做判断
        // 右子树的左子树和左子树的右子树做判断
        return func(left.right, right.left) && func(left.left, right.right);
    }
    
    
    
    /**
     * 机器人路径，回溯法
     * @return
     */
    public boolean hasPath(char[] matrix, int rows, int cols,char[] str) {
    	if (matrix == null || matrix.length <= 0)
    		return false;
    	boolean[] visited = new boolean[rows * cols];
    	for (int i = 0;i < rows;i++) {
    		for (int j = 0;j < cols;j++) {
    			if (hasPathCore(matrix, rows, cols, i, j, visited, 0, str)) {
    				return true;
    			}
    		}
    	}
    	return false;
    }
    
    public boolean hasPathCore(char[] matrix, int rows, int cols,int i, int j, boolean[] visited, int k,char[] str) {
		int index = i * cols + j;
		if (index >= rows * cols) return false;
		if (i > rows || i < 0 || j > cols || j < 0 || matrix[index] != str[k] || visited[index] == true)
			return false;
		if (k == str.length - 1) return true;
		
		visited[index] = true;
		boolean flag = hasPathCore(matrix, rows, cols, i + 1, j, visited, k + 1, str) ||
					hasPathCore(matrix, rows, cols, i - 1, j, visited, k + 1, str) ||
					hasPathCore(matrix, rows, cols, i, j + 1, visited, k + 1,  str) ||
					hasPathCore(matrix, rows, cols, i, j - 1, visited, k + 1, str);
		if (flag)
			return true;
		else {
			visited[index] = false;
			return false;
		}
		
	}

    public int movingCount(int threshold, int rows, int cols)
    {
    	boolean[] visited = new boolean[rows * cols];
    	return movingCountCore(threshold, rows, cols, 0, 0, visited);
    	
    }
    
    public int movingCountCore(int threshold, int rows, int cols, int i, int j, boolean[] visited) {
    	int index = i * cols + j;
    	if (index >= rows * cols) return 0;
    	if (i < 0 || i > rows || j < 0 || j > cols || visited[index] == true) {
    		return 0;
    	}
    	if (getSum(i, j) > threshold)
    		return 0;
    	
    	visited[index] = true;
    	return movingCountCore(threshold, rows, cols, i + 1, j, visited) +
    					movingCountCore(threshold, rows, cols, i - 1, j, visited) +
    					movingCountCore(threshold, rows, cols, i, j + 1, visited) +
    					movingCountCore(threshold, rows, cols, i, j-1, visited) + 1;
    }
    
    public int getSum(int i, int j) {
    	int sum1 = 0;
    	while (i != 0) {
    		sum1 += i % 10;
    		i = i / 10;
    	}
    	while (j != 0) {
    		sum1 += j % 10;
    		j = j / 10;
    	}
    	return sum1;
    }
    
    private static class ListNode {
    	int val;
    	ListNode next = null;
    	public ListNode(int val) {
    		this.val = val;
		}
    }
    
    
    class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }

    private static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
        TreeNode next = null;
        
        TreeNode(int val) {
            this.val = val;
        }
    }
}
