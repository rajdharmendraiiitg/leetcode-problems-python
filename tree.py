class Node:
	def __init__(self, key):
		self.left = None
		self.right = None
		self.val = key

class Node1:
	def __init__(self, key,left,right):
		self.left = left
		self.right = right
		self.val = key

def insert(root, key):
	if root is None:
		return Node(key)
	else:
		if root.val == key:
			return root
		elif root.val < key:
			root.right = insert(root.right, key)
		else:
			root.left = insert(root.left, key)
	return root

def inorder(root):
	if root:
		inorder(root.left)
		print(root.val)
		inorder(root.right)


def addBT(root):
    if (root == None):
        return 0
    return (root.val + addBT(root.left) + addBT(root.right))
def sumOfLeftLeaves(root): # using stack
        result = 0
        stack = [(root, False)]
        while stack:
            curr, is_left = stack.pop()
            #print(curr.val,is_left)
            if not curr:
                continue
            if not curr.left and not curr.right:
                if is_left:
                    result += curr.val
            else:
                stack.append((curr.left, True))
                stack.append((curr.right, False))
        return result
def sumOfLeftLeaves1(root): # using recursion
        if not root:
            return 0

        # does this node have a left child which is a leaf?
        if root.left and not root.left.left and not root.left.right:
			# gotcha
            return root.left.val + sumOfLeftLeaves1(root.right)

        # no it does not have a left child or it's not a leaf
        else:
			# bummer
            return sumOfLeftLeaves1(root.left) + sumOfLeftLeaves1(root.right)

def sumofleft(root): #use queue
    if root is None:
        return 0
    queue = []
    queue.append(root)
    sum = 0
    while(len(queue)>0):
        root=queue.pop(0)
        if root.left is not None:
            if root.left.left is None and root.left.right is None:
                sum += root.left.val
            else:
                queue.append(root.left)
        if root.right is not None:
            queue.append(root.right)
    return sum
def secondMin(root): # bfs
    ans = float('inf')
    queue = [root]
    while(len(queue)>0):
        node = queue.pop(0)
        if node.left is not None and node.right is not None:
            larger = max(node.left.val,node.right.val)
            if larger > node.val:
                ans = min(ans,larger)
            queue.append(node.left)
            queue.append(node.right)
    return ans if ans != float('inf') else -1
def isSymmetric(root): # will check if our root is none or not
    if root is None:
        return True
    def isMirror(leftroot,rightroot):
        if leftroot and rightroot: # both are not null
            return leftroot.val==rightroot.val and isMirror(leftroot.left,rightroot.right) and isMirror(leftroot.right,rightroot.left)
        return leftroot==rightroot
    return isMirror(root.left,root.right)
def levalOrderTrav(root):
    queue = [root]
    final_list = []
    if root is None: return final_list
    while(len(queue)>0):
        temp_list = []
        for i in range(len(queue)):
            node = queue.pop(0)
            temp_list.append(node.val)
            if node.left is not None: queue.append(node.left)
            if node.right is not None: queue.append(node.right)
        final_list.append(temp_list)
    return final_list
def levalZigZag(root):
    queue = [root]
    final_list = []
    if root is None:
        return final_list
    level_count = 0
    while(len(queue)>0):
        temp_list = [] # for even level
        stack = [] # for odd level
        for i in range(len(queue)):
            node = queue.pop(0)
            if level_count%2==0:
                temp_list.append(node.val)
            else:
                stack.append(node.val)
            if node.left is not None: queue.append(node.left)
            if node.right is not None: queue.append(node.right)
        while len(stack)!=0:
            temp_list.append(stack.pop())
        final_list.append(temp_list)
        level_count += 1
    return final_list
def maxDepth(root):
    if root is None: return 0
    return max(maxDepth(root.left),maxDepth(root.right))+1
    #return max(maxDepth(root.left,depth+1),maxDepth(root.right,depth+1))
def minDepth(root):
    if root is None: return 0
    return min(minDepth(root.left),minDepth(root.right))+1
    #return max(maxDepth(root.left,depth+1),maxDepth(root.right,depth+1))
def rightView(root):
    queue = [root]
    final_list = []
    if root is None: return final_list
    while(len(queue)>0):
        temp_list = []
        for i in range(len(queue)):
            node = queue.pop(0)
            temp_list.append(node.val)
            if node.left is not None: queue.append(node.left)
            if node.right is not None: queue.append(node.right)
        final_list.append(temp_list[-1])
    return final_list
def leftView(root):
    queue = [root]
    final_list = []
    if root is None: return final_list
    while(len(queue)>0):
        temp_list = []
        for i in range(len(queue)):
            node = queue.pop(0)
            temp_list.append(node.val)
            if node.left is not None: queue.append(node.left)
            if node.right is not None: queue.append(node.right)
        final_list.append(temp_list[0])
    return final_list
def flattenTreeInToList(root):
    if root is None:
        return None
    leftlist = flattenTreeInToList(root.left)
    rightlist = flattenTreeInToList(root.right)
    #print(root.val)
    if root.left is not None:
        leftlist.right = root.right # put root right to list next
        root.right=root.left # make root right lef
        root.left=None
    last = leftlist or rightlist or root
    return last

def deepestLeavesSum(root):
        queue = [root]
        final_list = []
        if root is None:
            return final_list
        while(len(queue)>0):
            temp_list = [] # for even level
            for i in range(len(queue)):
                node = queue.pop(0)
                temp_list.append(node.val)
                if node.left is not None: queue.append(node.left)
                if node.right is not None: queue.append(node.right)
            final_list.append(temp_list)
        return sum(final_list[-1])
def averageOfSubtree(root):
    count = 0
    def dfs(root):
        nonlocal count
        if root is None:
            return 0,0
        left_sum,left_node_count = dfs(root.left)
        right_sum,right_node_count = dfs(root.right)
        total_sum = root.val+left_sum+right_sum
        total_node_count = 1+left_node_count+right_node_count
        if total_sum//total_node_count==root.val:
            count += 1
        return total_sum,total_node_count
    dfs(root)
    return count
def sumEvenGrandparent(root):
        nodes_and_preds = []
        nodes_and_preds.append((root, None, None)) # in this order=> current_node, parent, grandparent
        answer=0
        while len(nodes_and_preds)>0:
            cur_node, parent, grandpar = nodes_and_preds.pop() # use pop(0) to make using queue
            if parent and grandpar and grandpar.val%2==0:
                answer+=cur_node.val
            if cur_node.left:
                nodes_and_preds.append((cur_node.left, cur_node, parent))
            if cur_node.right:
                nodes_and_preds.append((cur_node.right, cur_node, parent))
        return answer
def bstToGst(root):
    node_sum = 0
    def rInorder(root):
        nonlocal node_sum
        if root is None:
            return None
        else:
            rInorder(root.right)
            node_sum += root.val
            root.val = node_sum
            rInorder(root.left)
            print(root.val)
            return root
    return node_sum
def removeLeafNodes(root,target):
    if root:
        root.left = removeLeafNodes(root.left, target)
        root.right = removeLeafNodes(root.right, target)
        if root.val == target and root.left is root.right:
            return None
    return root
def trimBST(root,low,high):
    '''if root is None:
        return None
    if root.val<low:
        return trimBST(root.right,low,high)
    if root.val>high:
        return trimBST(root.left.low.high)
    root.left = trimBST(root.left.low.high)
    root.right= trimBST(root.righ,low,high)'''
    if root is None:
        return None
    else:
        if root.val>= low and root.val<=high:
            root.left = trimBST(root.left.low.high)
            root.right= trimBST(root.righ,low,high)
            return root
        elif root.val<low:
            return trimBST(root.right,low,high)
        elif root.val>high:
            return trimBST(root.left.low.high)
    return root
def findBottomLeftValue(root):
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.right is not None: queue.append(node.right)
        if node.left is not None: queue.append(node.left)
    return node.val
def allPossibleFBT(n):
    if n==0:
        return []
    if n==1:
        return [Node1(0,None,None)]
    result = []
    for left in range(n):
        right = n-left-1
        lefttrees,righttrees = allPossibleFBT(left),allPossibleFBT(right)
        for tLeft in lefttrees:
            for tRight in righttrees:
                result.append(Node1(0,tLeft,tRight))
    return result
def rob(root):
    def dfs(root): # return pair of values: [withroot,withoutroot]
        if root is None:
            return [0,0]
        leftpair = dfs(root.left)
        rightpair = dfs(root.right)
        withroot = root.val + leftpair[1] + rightpair[1]
        withoutroot = max(leftpair) + max(rightpair)
        return [withroot,withoutroot]
    return max(dfs(root))
def isValidBinary(root):
    def valid(node,left,right):
            if not node:
                return True
            if not (node.val > left and node.val < right):
                return False
            return (valid(node.left,left,node.val) and valid(node.right,node.val,right))
    return valid(root,float("-inf"),float("inf"))
def atKdistance(root):
    parentsChildMap = {}
    parentMap = {}
    def parentsChild(parentsChildMap ,root):
        if root is None: return
        if root.left is not None:
            parentsChildMap[root.left.val]=root.val
        if root.right is not None:
            parentsChildMap[root.right.val]=root.val
        parentsChild(parentsChildMap,root.left)
        parentsChild(parentsChildMap,root.right)
    #parentsChild(parentsChildMap, root)
    def buildParentMap(node, parent, parentMap):
        if node is None:
            return
        #parentMap[node] = parent
        if parent is not None:
            parentMap[node.val] = parent.val
        buildParentMap(node.left, node, parentMap)
        buildParentMap(node.right, node, parentMap)
    buildParentMap(root, None, parentMap)
    return parentMap
# Driver program to test the above functions
# Let us create the following BST
#        50
#      /    \
#     30     70
#    / \    /   \
#  20   40  60   80

root = Node(50)
nodes_list = [30,20,40,70,60,80]
for node in nodes_list:
    root=insert(root,node)
#inorder(root)
#print(addBT(root))
#print(sumOfLeftLeaves1(root))
#print(sumofleft(root))
#print(secondMin(root))
#print(isSymmetric(root))
#print(levalOrderTrav(root))
#print(levalZigZag(root))
#print(maxDepth(root))
#print(rightView(root))
#print(leftView(root))
#print(flattenTreeInToList(root))
#print(deepestLeavesSum(root))
#print(averageOfSubtree(root))
#print(sumEvenGrandparent(root))
#print(bstToGst(root,20))
#print(removeLeafNodes(root,20))
#print(trimBST(root,low,high))
#print(findBottomLeftValue(root))
#print(allPossibleFBT(7))
#print(rob(root))
#print(isValidBinary(root))
print(atKdistance(root))