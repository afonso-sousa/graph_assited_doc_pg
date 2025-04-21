from .glossary import UPOS


class LinearizationMixin:
    def find_roots(self):
        if not self.edges:
            return {0}
        indexes = set()
        for tup in self.edges:
            indexes.add(tup[0])
        for tup in self.edges:
            if tup[1] in indexes:
                indexes.remove(tup[1])

        if indexes == set():
            return {0}

        return sorted(indexes)

    def traverse(
        self,
        node,
        visited,
        reentrancy_tokens=False,
        delimiters=False,
        relations=False,
    ):
        node_idx, node_obj, relation = node

        result = []
        # Check if the node has been visited
        if node_idx in visited:
            return result

        # Add the current node to the visited list
        visited.append(node_idx)

        children = self.get_children(node_idx)
        node_token = f"<R{node_idx}>"
        relation_str = f":{relation}" if relation else ""

        children_lists = []
        for child in children:
            children_lists += self.traverse(
                child,
                visited,
                reentrancy_tokens=reentrancy_tokens,
                delimiters=delimiters,
                relations=relations,
            )

        if relation_str:
            result += [(2, relation_str)] if relations else []
        result += (
            ([(1, "<L>")] if delimiters else [])
            + ([(1, node_token)] if reentrancy_tokens else [])
            + [
                (list(UPOS.keys()).index(node_obj.ud_pos) + 3, word)
                for word in node_obj.word
            ]
            + children_lists
            + ([(1, "<R>")] if delimiters else [])
        )
        return result

    def linearize(
        self,
        reentrancy_tokens=False,
        delimiters=False,
        relations=False,
    ):
        roots = self.find_roots()
        token_list = []
        for root_idx in roots:
            token_list += self.traverse(
                (root_idx, self.nodes[root_idx], ""),
                [],
                reentrancy_tokens=reentrancy_tokens,
                delimiters=delimiters,
                relations=relations,
            )

        return token_list
