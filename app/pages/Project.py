import streamlit as st
st.title('Project Page')

st.image("app/io/Project.png")

st.latex(r'''
\begin{equation}
\int\left(\prod_{i=1}^{N_{T+1}} p\left(r_{i, T+1} \mid \mathbf{F}_{\mathbf{T}+1}, \mathcal{F}_T\right)\right) p\left(\mathbf{F}_{\mathbf{T}+1} \mid \mathcal{F}_T\right) d \mathbf{F}_{\mathbf{T}+1}
\end{equation}
''')